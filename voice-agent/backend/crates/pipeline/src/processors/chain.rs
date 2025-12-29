//! Channel-based processor chain
//!
//! Connects multiple FrameProcessors with tokio channels for
//! concurrent, streaming frame processing.

use std::sync::Arc;
use tokio::sync::mpsc;
use voice_agent_core::{Frame, FrameProcessor, ProcessorContext, Result};

/// Channel capacity for inter-processor communication
const DEFAULT_CHANNEL_CAPACITY: usize = 64;

/// A chain of frame processors connected by channels
pub struct ProcessorChain {
    /// Name of this chain
    name: String,
    /// Processors in order
    processors: Vec<Arc<dyn FrameProcessor>>,
    /// Channel capacity
    channel_capacity: usize,
}

impl ProcessorChain {
    /// Create a new processor chain
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            processors: Vec::new(),
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
        }
    }

    /// Create using the builder
    pub fn builder(name: impl Into<String>) -> ProcessorChainBuilder {
        ProcessorChainBuilder::new(name)
    }

    /// Add a processor to the chain
    pub fn add<P: FrameProcessor + 'static>(&mut self, processor: P) -> &mut Self {
        self.processors.push(Arc::new(processor));
        self
    }

    /// Add a boxed processor to the chain
    pub fn add_boxed(&mut self, processor: Arc<dyn FrameProcessor>) -> &mut Self {
        self.processors.push(processor);
        self
    }

    /// Get the chain name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get number of processors
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Process a single frame through the chain synchronously
    ///
    /// This processes frames one by one through each processor.
    /// For streaming, use `run` instead.
    pub async fn process_one(
        &self,
        frame: Frame,
        context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        let mut frames = vec![frame];

        for processor in &self.processors {
            let mut next_frames = Vec::new();

            for f in frames {
                let output = processor.process(f, context).await?;
                next_frames.extend(output);
            }

            frames = next_frames;
        }

        Ok(frames)
    }

    /// Start the processing pipeline
    ///
    /// Spawns a task for each processor connected by channels.
    /// Returns the input sender and output receiver.
    pub fn run(
        &self,
        initial_context: ProcessorContext,
    ) -> (mpsc::Sender<Frame>, mpsc::Receiver<Frame>) {
        let (input_tx, input_rx) = mpsc::channel::<Frame>(self.channel_capacity);

        if self.processors.is_empty() {
            // Empty chain: directly connect input to output
            let (output_tx, output_rx) = mpsc::channel::<Frame>(self.channel_capacity);
            let mut input_rx = input_rx;

            tokio::spawn(async move {
                while let Some(frame) = input_rx.recv().await {
                    if output_tx.send(frame).await.is_err() {
                        break;
                    }
                }
            });

            return (input_tx, output_rx);
        }

        // Create channels between processors
        let mut current_rx = input_rx;
        let mut final_tx = None;

        for (i, processor) in self.processors.iter().enumerate() {
            let is_last = i == self.processors.len() - 1;
            let processor = Arc::clone(processor);
            let mut context = initial_context.clone();

            let (next_tx, next_rx) = mpsc::channel::<Frame>(self.channel_capacity);

            let rx = current_rx;
            let tx = next_tx.clone();
            let processor_name = processor.name().to_string();

            // Spawn processor task
            tokio::spawn(async move {
                // Call on_start
                if let Err(e) = processor.on_start(&mut context).await {
                    tracing::error!(
                        processor = processor_name,
                        error = %e,
                        "Processor on_start failed"
                    );
                }

                let mut rx = rx;

                while let Some(frame) = rx.recv().await {
                    let is_eos = frame.is_end_of_stream();

                    match processor.process(frame, &mut context).await {
                        Ok(output_frames) => {
                            for output_frame in output_frames {
                                if tx.send(output_frame).await.is_err() {
                                    tracing::debug!(
                                        processor = processor_name,
                                        "Output channel closed"
                                    );
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                processor = processor_name,
                                error = %e,
                                "Processor error"
                            );
                            // Send error frame
                            let error_frame = Frame::Error {
                                stage: processor_name.clone(),
                                message: e.to_string(),
                                recoverable: true,
                            };
                            let _ = tx.send(error_frame).await;
                        }
                    }

                    if is_eos {
                        // Call on_stop on EOS
                        if let Err(e) = processor.on_stop(&mut context).await {
                            tracing::error!(
                                processor = processor_name,
                                error = %e,
                                "Processor on_stop failed"
                            );
                        }
                    }
                }

                tracing::debug!(processor = processor_name, "Processor task exiting");
            });

            if is_last {
                final_tx = Some(next_tx);
            }

            current_rx = next_rx;
        }

        // final_tx is not used; the last processor writes to current_rx
        drop(final_tx);

        (input_tx, current_rx)
    }

    /// Run with a custom output handler
    ///
    /// Calls the handler for each output frame.
    pub async fn run_with_handler<F, Fut>(
        &self,
        initial_context: ProcessorContext,
        input: mpsc::Receiver<Frame>,
        mut handler: F,
    ) -> Result<()>
    where
        F: FnMut(Frame) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        let (tx, mut output) = self.run(initial_context);

        // Forward input to chain
        let mut input = input;
        let forward_task = tokio::spawn(async move {
            while let Some(frame) = input.recv().await {
                if tx.send(frame).await.is_err() {
                    break;
                }
            }
        });

        // Handle output
        while let Some(frame) = output.recv().await {
            handler(frame).await?;
        }

        // Wait for forward task
        let _ = forward_task.await;

        Ok(())
    }
}

/// Builder for ProcessorChain
pub struct ProcessorChainBuilder {
    chain: ProcessorChain,
}

impl ProcessorChainBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            chain: ProcessorChain::new(name),
        }
    }

    /// Add a processor
    pub fn processor<P: FrameProcessor + 'static>(mut self, processor: P) -> Self {
        self.chain.add(processor);
        self
    }

    /// Add a boxed processor
    pub fn processor_boxed(mut self, processor: Arc<dyn FrameProcessor>) -> Self {
        self.chain.add_boxed(processor);
        self
    }

    /// Set channel capacity
    pub fn channel_capacity(mut self, capacity: usize) -> Self {
        self.chain.channel_capacity = capacity;
        self
    }

    /// Build the chain
    pub fn build(self) -> ProcessorChain {
        self.chain
    }
}

/// A passthrough processor for testing
pub struct PassthroughProcessor {
    name: &'static str,
}

impl PassthroughProcessor {
    /// Create a new passthrough processor
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

#[async_trait::async_trait]
impl FrameProcessor for PassthroughProcessor {
    async fn process(
        &self,
        frame: Frame,
        _context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        Ok(vec![frame])
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

/// A filter processor that only passes frames matching a predicate
pub struct FilterProcessor<F>
where
    F: Fn(&Frame) -> bool + Send + Sync + 'static,
{
    name: &'static str,
    predicate: F,
}

impl<F> FilterProcessor<F>
where
    F: Fn(&Frame) -> bool + Send + Sync + 'static,
{
    /// Create a new filter processor
    pub fn new(name: &'static str, predicate: F) -> Self {
        Self { name, predicate }
    }
}

#[async_trait::async_trait]
impl<F> FrameProcessor for FilterProcessor<F>
where
    F: Fn(&Frame) -> bool + Send + Sync + 'static,
{
    async fn process(
        &self,
        frame: Frame,
        _context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        if (self.predicate)(&frame) {
            Ok(vec![frame])
        } else {
            Ok(vec![])
        }
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

/// A map processor that transforms frames
pub struct MapProcessor<F>
where
    F: Fn(Frame) -> Frame + Send + Sync + 'static,
{
    name: &'static str,
    mapper: F,
}

impl<F> MapProcessor<F>
where
    F: Fn(Frame) -> Frame + Send + Sync + 'static,
{
    /// Create a new map processor
    pub fn new(name: &'static str, mapper: F) -> Self {
        Self { name, mapper }
    }
}

#[async_trait::async_trait]
impl<F> FrameProcessor for MapProcessor<F>
where
    F: Fn(Frame) -> Frame + Send + Sync + 'static,
{
    async fn process(
        &self,
        frame: Frame,
        _context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        Ok(vec![(self.mapper)(frame)])
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voice_agent_core::Language;

    #[tokio::test]
    async fn test_empty_chain() {
        let chain = ProcessorChain::new("empty");
        let mut ctx = ProcessorContext::default();

        let frames = chain
            .process_one(Frame::VoiceStart, &mut ctx)
            .await
            .unwrap();

        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::VoiceStart));
    }

    #[tokio::test]
    async fn test_passthrough_chain() {
        let chain = ProcessorChain::builder("test")
            .processor(PassthroughProcessor::new("p1"))
            .processor(PassthroughProcessor::new("p2"))
            .build();

        let mut ctx = ProcessorContext::default();

        let frames = chain
            .process_one(Frame::VoiceStart, &mut ctx)
            .await
            .unwrap();

        assert_eq!(frames.len(), 1);
    }

    #[tokio::test]
    async fn test_filter_chain() {
        let chain = ProcessorChain::builder("filter")
            .processor(FilterProcessor::new("filter", |f| {
                !matches!(f, Frame::VoiceStart)
            }))
            .build();

        let mut ctx = ProcessorContext::default();

        // VoiceStart should be filtered out
        let frames = chain
            .process_one(Frame::VoiceStart, &mut ctx)
            .await
            .unwrap();
        assert!(frames.is_empty());

        // VoiceEnd should pass through
        let frames = chain
            .process_one(Frame::VoiceEnd { duration_ms: 100 }, &mut ctx)
            .await
            .unwrap();
        assert_eq!(frames.len(), 1);
    }

    #[tokio::test]
    async fn test_running_chain() {
        let chain = ProcessorChain::builder("running")
            .processor(PassthroughProcessor::new("p1"))
            .channel_capacity(16)
            .build();

        let ctx = ProcessorContext::new("test-session");
        let (tx, mut rx) = chain.run(ctx);

        // Send frames
        tx.send(Frame::VoiceStart).await.unwrap();
        tx.send(Frame::Sentence {
            text: "Hello".into(),
            language: Language::English,
            index: 0,
        })
        .await
        .unwrap();
        tx.send(Frame::EndOfStream).await.unwrap();

        // Receive frames
        let mut received = Vec::new();
        while let Some(frame) = rx.recv().await {
            let is_eos = frame.is_end_of_stream();
            received.push(frame);
            if is_eos {
                break;
            }
        }

        assert_eq!(received.len(), 3);
    }

    #[tokio::test]
    async fn test_chain_with_sentence_detector() {
        use super::super::SentenceDetector;

        let chain = ProcessorChain::builder("with_detector")
            .processor(SentenceDetector::default_config())
            .build();

        let mut ctx = ProcessorContext::default();

        let frames = chain
            .process_one(
                Frame::LLMChunk {
                    text: "Hello world. How are you?".into(),
                    is_final: true,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should have sentences
        let sentence_count = frames
            .iter()
            .filter(|f| matches!(f, Frame::Sentence { .. }))
            .count();

        assert!(sentence_count >= 1);
    }

    #[tokio::test]
    async fn test_builder() {
        let chain = ProcessorChain::builder("builder_test")
            .channel_capacity(32)
            .processor(PassthroughProcessor::new("test"))
            .build();

        assert_eq!(chain.name(), "builder_test");
        assert_eq!(chain.len(), 1);
    }
}
