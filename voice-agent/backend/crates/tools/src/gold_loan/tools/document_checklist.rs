//! Document Checklist Tool
//!
//! Get the list of documents required for gold loan application.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Document checklist tool
pub struct DocumentChecklistTool;

impl DocumentChecklistTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DocumentChecklistTool {
    fn name(&self) -> &str {
        "get_document_checklist"
    }

    fn description(&self) -> &str {
        "Get the list of documents required for gold loan application based on loan type and customer category"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "loan_type",
                    PropertySchema::enum_type(
                        "Type of gold loan",
                        vec![
                            "new_loan".into(),
                            "top_up".into(),
                            "balance_transfer".into(),
                            "renewal".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "customer_type",
                    PropertySchema::enum_type(
                        "Customer category",
                        vec![
                            "individual".into(),
                            "self_employed".into(),
                            "business".into(),
                            "nri".into(),
                        ],
                    ),
                    false,
                )
                .property(
                    "existing_customer",
                    PropertySchema::boolean("Is an existing customer"),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let loan_type = input
            .get("loan_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("loan_type is required"))?;

        let customer_type = input
            .get("customer_type")
            .and_then(|v| v.as_str())
            .unwrap_or("individual");

        let existing_customer = input
            .get("existing_customer")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // TODO: Document requirements should come from domain config
        let mut mandatory_docs = vec![
            json!({
                "document": "Valid Photo ID",
                "accepted": ["Aadhaar Card", "PAN Card", "Passport", "Voter ID", "Driving License"],
                "copies": 1,
                "notes": "Original required for verification"
            }),
            json!({
                "document": "Address Proof",
                "accepted": ["Aadhaar Card", "Utility Bill (last 3 months)", "Bank Statement", "Rent Agreement"],
                "copies": 1,
                "notes": "Should match current residence"
            }),
            json!({
                "document": "Passport Size Photographs",
                "copies": 2,
                "notes": "Recent photographs (within 6 months)"
            }),
        ];

        mandatory_docs.push(json!({
            "document": "PAN Card",
            "copies": 1,
            "notes": "Mandatory for loans above ₹50,000"
        }));

        let gold_docs = vec![
            json!({
                "document": "Gold Items",
                "notes": "Bring gold jewelry/items for valuation. Remove any non-gold attachments (stones, pearls)"
            }),
            json!({
                "document": "Gold Purchase Invoice (if available)",
                "notes": "Helps with valuation and authenticity verification"
            }),
        ];

        let additional_docs: Vec<Value> = match loan_type {
            "balance_transfer" => vec![
                json!({
                    "document": "Existing Loan Statement",
                    "notes": "From current lender showing outstanding amount"
                }),
                json!({
                    "document": "Gold Loan Account Details",
                    "notes": "Loan account number and lender details"
                }),
                json!({
                    "document": "NOC from Current Lender",
                    "notes": "May be obtained after approval"
                }),
            ],
            "top_up" => vec![json!({
                "document": "Existing Gold Loan Details",
                "notes": "Loan account number for top-up"
            })],
            "renewal" => vec![json!({
                "document": "Previous Loan Details",
                "notes": "Loan account number for renewal"
            })],
            _ => vec![],
        };

        let customer_specific: Vec<Value> = match customer_type {
            "self_employed" | "business" => vec![json!({
                "document": "Business Proof",
                "accepted": ["GST Registration", "Shop & Establishment Certificate", "Trade License"],
                "notes": "Any one document for business verification"
            })],
            "nri" => vec![
                json!({
                    "document": "Passport with Valid Visa",
                    "notes": "Required for NRI customers"
                }),
                json!({
                    "document": "NRE/NRO Bank Account Statement",
                    "notes": "Last 6 months statement"
                }),
            ],
            _ => vec![],
        };

        let existing_customer_note = if existing_customer {
            "As an existing customer, some documents may already be on file. Please bring originals for verification."
        } else {
            "Please bring original documents along with photocopies."
        };

        let result = json!({
            "loan_type": loan_type,
            "customer_type": customer_type,
            "existing_customer": existing_customer,
            "mandatory_documents": mandatory_docs,
            "gold_related": gold_docs,
            "additional_documents": additional_docs,
            "customer_specific_documents": customer_specific,
            "total_documents": mandatory_docs.len() + gold_docs.len() + additional_docs.len() + customer_specific.len(),
            "important_notes": [
                existing_customer_note,
                "Original documents are required for verification at the branch.",
                "Gold items should be free of non-gold attachments for accurate valuation.",
                "Processing time: Same day disbursement subject to document verification."
            ],
            "message": format!(
                "For a {} gold loan, you'll need {} documents. Key documents: Valid ID, Address Proof, PAN Card, and your gold items.",
                loan_type.replace("_", " "),
                mandatory_docs.len() + gold_docs.len() + additional_docs.len() + customer_specific.len()
            )
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for DocumentChecklistTool {
    fn default() -> Self {
        Self::new()
    }
}
