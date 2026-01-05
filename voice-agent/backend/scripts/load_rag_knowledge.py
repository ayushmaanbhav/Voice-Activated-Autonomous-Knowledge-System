#!/usr/bin/env python3
"""Load gold loan knowledge into Qdrant for RAG."""

import json
import os
import requests
import uuid
import yaml

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "gold_loan_knowledge"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")

print(f"Using Ollama embedding model: {EMBEDDING_MODEL}")

def generate_embedding(text: str) -> list:
    """Generate embedding using Ollama Qwen3."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text}
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def upsert_point(point_id: str, vector: list, payload: dict):
    """Insert point into Qdrant."""
    # Use UUID string format for Qdrant compatibility
    qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
        json={
            "points": [{
                "id": qdrant_id,
                "vector": vector,
                "payload": payload
            }]
        }
    )
    if resp.status_code != 200:
        print(f"  Error upserting {point_id}: {resp.text}")
    return resp.json()

def load_branches():
    """Load branch information."""
    print("\nLoading branches...")
    with open("data/branches.json") as f:
        data = json.load(f)

    for branch in data["branches"]:
        # Create searchable text
        text = f"""
        Kotak Mahindra Bank branch at {branch['area']}, {branch['city']}.
        Address: {branch['address']}
        Phone: {branch['phone']}
        Timing: {branch['timing']}
        Facilities: {', '.join(branch['facilities'])}
        Gold loan available: {'Yes' if branch['gold_loan_available'] else 'No'}
        """

        # Also add Hindi version
        text_hi = f"""
        कोटक महिंद्रा बैंक शाखा {branch['area']}, {branch['city']} में।
        पता: {branch['address']}
        फोन: {branch['phone']}
        समय: {branch['timing']}
        सुविधाएं: {', '.join(branch['facilities'])}
        गोल्ड लोन उपलब्ध: {'हाँ' if branch['gold_loan_available'] else 'नहीं'}
        """

        combined_text = text + "\n" + text_hi
        vector = generate_embedding(combined_text)

        payload = {
            "text": combined_text,
            "type": "branch",
            "city": branch["city"],
            "area": branch["area"],
            "branch_id": branch["branch_id"]
        }

        result = upsert_point(branch["branch_id"], vector, payload)
        print(f"  Loaded branch: {branch['name']}")

    print(f"Loaded {len(data['branches'])} branches")

def load_gold_loan_info():
    """Load gold loan product information."""
    print("\nLoading gold loan information...")

    documents = [
        # Interest rates
        {
            "id": "GL001",
            "text": """Kotak Mahindra Bank Gold Loan Interest Rates:
            - Standard rate: 10.5% per annum
            - For loans up to ₹1 lakh: 11.5% per annum
            - For loans ₹1 lakh to ₹5 lakh: 10.5% per annum
            - For loans above ₹5 lakh: 9.5% per annum
            Competitor comparison: Muthoot Finance charges 18%, Manappuram 19%, IIFL 17.5%
            Kotak offers the best rates in the market!

            कोटक महिंद्रा बैंक गोल्ड लोन ब्याज दरें:
            - मानक दर: 10.5% प्रति वर्ष
            - ₹1 लाख तक के लोन के लिए: 11.5% प्रति वर्ष
            - ₹1 लाख से ₹5 लाख तक: 10.5% प्रति वर्ष
            - ₹5 लाख से अधिक: 9.5% प्रति वर्ष""",
            "type": "product",
            "topic": "interest_rates"
        },
        # Loan amount
        {
            "id": "GL002",
            "text": """Gold Loan Amount and LTV:
            - Minimum loan: ₹10,000
            - Maximum loan: ₹2.5 crore
            - Loan-to-Value (LTV): Up to 75% of gold value
            - Gold price: Approximately ₹7,500 per gram (varies daily)

            गोल्ड लोन राशि और LTV:
            - न्यूनतम लोन: ₹10,000
            - अधिकतम लोन: ₹2.5 करोड़
            - लोन-टू-वैल्यू (LTV): सोने के मूल्य का 75% तक
            - सोने की कीमत: लगभग ₹7,500 प्रति ग्राम""",
            "type": "product",
            "topic": "loan_amount"
        },
        # Processing fee
        {
            "id": "GL003",
            "text": """Gold Loan Processing Fee and Charges:
            - Processing fee: 1% of loan amount
            - No prepayment penalty
            - No foreclosure charges after 3 months
            - Documentation charges: Nil
            - Valuation charges: Free

            गोल्ड लोन प्रोसेसिंग फीस:
            - प्रोसेसिंग फीस: लोन राशि का 1%
            - कोई प्रीपेमेंट पेनल्टी नहीं
            - 3 महीने बाद कोई फोरक्लोज़र चार्ज नहीं
            - डॉक्यूमेंटेशन चार्ज: शून्य
            - वैल्यूएशन चार्ज: मुफ्त""",
            "type": "product",
            "topic": "fees"
        },
        # Documents required
        {
            "id": "GL004",
            "text": """Documents Required for Gold Loan:
            1. ID Proof: Aadhaar Card, PAN Card, Passport, or Voter ID
            2. Address Proof: Aadhaar Card, Passport, Utility Bills
            3. Passport size photographs (2)
            4. Gold ornaments for pledging
            That's all! Minimal documentation for quick processing.

            गोल्ड लोन के लिए आवश्यक दस्तावेज:
            1. पहचान प्रमाण: आधार कार्ड, पैन कार्ड, पासपोर्ट, या वोटर आईडी
            2. पता प्रमाण: आधार कार्ड, पासपोर्ट, बिजली/गैस बिल
            3. पासपोर्ट साइज फोटो (2)
            4. गिरवी रखने के लिए सोने के गहने""",
            "type": "product",
            "topic": "documents"
        },
        # Eligibility
        {
            "id": "GL005",
            "text": """Gold Loan Eligibility:
            - Age: 18 years and above
            - Indian resident
            - Gold purity: 18 karat or above accepted
            - No income proof required
            - No credit score check
            - Self-employed, salaried, or homemakers - everyone eligible!

            गोल्ड लोन पात्रता:
            - आयु: 18 वर्ष और उससे अधिक
            - भारतीय निवासी
            - सोने की शुद्धता: 18 कैरेट या उससे अधिक
            - आय प्रमाण आवश्यक नहीं
            - क्रेडिट स्कोर जांच नहीं
            - स्व-रोजगार, वेतनभोगी, या गृहिणी - सभी पात्र!""",
            "type": "product",
            "topic": "eligibility"
        },
        # Tenure
        {
            "id": "GL006",
            "text": """Gold Loan Tenure Options:
            - Minimum tenure: 3 months
            - Maximum tenure: 36 months (3 years)
            - Flexible repayment options:
              * Pay interest monthly, principal at end
              * EMI option available
              * Bullet payment (interest + principal at end)

            गोल्ड लोन अवधि विकल्प:
            - न्यूनतम अवधि: 3 महीने
            - अधिकतम अवधि: 36 महीने (3 वर्ष)
            - लचीले पुनर्भुगतान विकल्प:
              * मासिक ब्याज भुगतान, अंत में मूलधन
              * EMI विकल्प उपलब्ध
              * बुलेट भुगतान (अंत में ब्याज + मूलधन)""",
            "type": "product",
            "topic": "tenure"
        },
        # Process
        {
            "id": "GL007",
            "text": """Gold Loan Process - Quick and Easy:
            1. Visit nearest Kotak branch with gold and documents
            2. Gold valuation by certified appraiser (free)
            3. Fill application form
            4. Loan approval in 30 minutes
            5. Amount disbursed to your account same day!

            Doorstep service available in select cities.

            गोल्ड लोन प्रक्रिया - त्वरित और आसान:
            1. नज़दीकी कोटक शाखा में सोना और दस्तावेज़ लेकर जाएं
            2. प्रमाणित मूल्यांकनकर्ता द्वारा सोने का मूल्यांकन (मुफ्त)
            3. आवेदन फॉर्म भरें
            4. 30 मिनट में लोन स्वीकृति
            5. उसी दिन राशि आपके खाते में!""",
            "type": "product",
            "topic": "process"
        },
        # Why Kotak
        {
            "id": "GL008",
            "text": """Why Choose Kotak Gold Loan?
            - Lowest interest rates starting at 10.5%
            - Quick disbursement in 30 minutes
            - Safe storage in bank vault
            - Insurance coverage for pledged gold
            - Transparent valuation process
            - No hidden charges
            - Easy online tracking of loan

            कोटक गोल्ड लोन क्यों चुनें?
            - सबसे कम ब्याज दर 10.5% से शुरू
            - 30 मिनट में त्वरित वितरण
            - बैंक वॉल्ट में सुरक्षित भंडारण
            - गिरवी रखे सोने के लिए बीमा कवर
            - पारदर्शी मूल्यांकन प्रक्रिया
            - कोई छिपे हुए शुल्क नहीं
            - लोन की आसान ऑनलाइन ट्रैकिंग""",
            "type": "product",
            "topic": "benefits"
        }
    ]

    for doc in documents:
        vector = generate_embedding(doc["text"])
        payload = {
            "text": doc["text"],
            "type": doc["type"],
            "topic": doc["topic"]
        }
        result = upsert_point(doc["id"], vector, payload)
        print(f"  Loaded: {doc['topic']}")

    print(f"Loaded {len(documents)} product documents")

def load_yaml_knowledge():
    """Load knowledge from YAML files in knowledge/ directory."""
    print("\nLoading YAML knowledge files...")

    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"  Knowledge directory not found: {KNOWLEDGE_DIR}")
        return 0

    total_loaded = 0
    yaml_files = [f for f in os.listdir(KNOWLEDGE_DIR)
                  if f.endswith('.yaml') and f != 'manifest.yaml']

    for fname in sorted(yaml_files):
        fpath = os.path.join(KNOWLEDGE_DIR, fname)
        try:
            with open(fpath) as f:
                data = yaml.safe_load(f)

            docs = data.get('documents', [])
            if not docs:
                continue

            for doc in docs:
                doc_id = doc.get('id', '')
                title = doc.get('title', '')
                content = doc.get('content', '').strip()
                category = doc.get('category', 'general')
                language = doc.get('language', 'en')
                keywords = doc.get('keywords', [])
                segment_relevance = doc.get('segment_relevance', [])

                # Create combined text for embedding
                text = f"{title}\n\n{content}"

                # Generate embedding
                try:
                    vector = generate_embedding(text)
                except Exception as e:
                    print(f"    Error embedding {doc_id}: {e}")
                    continue

                # Create payload
                payload = {
                    "id": doc_id,
                    "text": text,
                    "title": title,
                    "content": content,
                    "type": category,
                    "category": category,
                    "language": language,
                    "keywords": keywords,
                    "segment_relevance": segment_relevance,
                    "source_file": fname
                }

                result = upsert_point(doc_id, vector, payload)
                total_loaded += 1

            print(f"  Loaded {len(docs)} docs from {fname}")

        except Exception as e:
            print(f"  Error loading {fname}: {e}")

    print(f"Total YAML documents loaded: {total_loaded}")
    return total_loaded


def clear_collection():
    """Clear all points from the collection."""
    print("\nClearing existing collection...")
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/delete",
        json={"filter": {}}  # Empty filter = delete all
    )
    if resp.status_code == 200:
        print("  Collection cleared")
    else:
        print(f"  Warning: Could not clear collection: {resp.text}")


def main():
    print("="*60)
    print("Loading Gold Loan Knowledge into Qdrant")
    print("="*60)

    # Verify collection exists
    resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    if resp.status_code != 200:
        print(f"Error: Collection {COLLECTION_NAME} not found!")
        return

    print(f"Collection '{COLLECTION_NAME}' found")

    # Clear existing data
    clear_collection()

    # Load all knowledge
    load_branches()
    load_gold_loan_info()
    load_yaml_knowledge()

    # Verify
    resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    info = resp.json()
    print(f"\nTotal points in collection: {info['result']['points_count']}")
    print("\nKnowledge base loaded successfully!")

if __name__ == "__main__":
    main()
