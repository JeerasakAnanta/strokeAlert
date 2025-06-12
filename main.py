import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from docling.document_converter import DocumentConverter

# ---------- Step 1: Convert Document ----------
source = "data/STROKEbyYear2025.xlsx"
converter = DocumentConverter()
result = converter.convert(source)

# Export to markdown
markdown_text = result.document.export_to_markdown()

# Save to file
with open("dataset.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)

# ---------- Step 2: Load Environment Variables ----------
load_dotenv(".env")

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_ACCESS_TOKEN"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Step 3: Convert Markdown to LangChain Documents ----------
content_list = markdown_text.split("\n")
langchain_documents = [
    Document(page_content=content.strip())
    for content in content_list
    if content.strip()
]

# ---------- Step 4: Create Vector Store ----------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(langchain_documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})
llm = ChatOpenAI(model="gpt-4o-mini")

# ---------- Step 5: Define Prompt Template ----------
template = """คุณคือผู้ช่วย อาสาสมัครสาธารณสุขประจำหมู่บ้าน (อสม.) 
ข้อมูลสุขภาพของผู้ป่วยคุณสามารถตอบคำถามเกี่ยวกับข้อมูลสุขภาพของผู้ป่วยได้ 
โดยข้อมูลมี Field อยู่ในฐานข้อมูลมีดังนี้  
hn               -- หมายเลขผู้ป่วย,
date             -- วันที่บันทึกข้อมูล,
full_name        -- ชื่อและนามสกุลของผู้ป่วย,
address          -- ที่อยู่ของผู้ป่วย,
bmi              -- ดัชนีมวลกาย (Body Mass Index),
sbp              -- ความดันโลหิตซิสโตลิก (Systolic Blood Pressure),
dbp              -- ความดันโลหิตด้านล่าง (Diastolic Blood Pressure),
blood_sugar      -- ระดับน้ำตาลในเลือด,
bone_status      -- สถานะของกระดูก,
dementia         -- สถานะสมองเสื่อม     (0 = เท็จ, 1 = จริง),
depression       -- สถานะซึมเศร้า       (0 = เท็จ, 1 = จริง)
nutrition_status -- สถานะโภชนาการ,
smoking          -- สถานะการสูบบุหรี่     (0 = เท็จ, 1 = จริง),
recorder         -- ผู้บันทึกข้อมูล,

ตอบโดยใช้ภาษาไทย และ มี emoji ประกอบการตอบคำถาม
ตัวอย่างการตอบคำถาม เช่น
"ผู้ป่วยหมายเลข 123456789 ชื่อ นาย ก มีความดันโลหิต 120/80 mmHg น้ำตาลในเลือด 100 mg/dL และ มีดัชนีมวลกาย 22.5 kg/m² ข้อมูลวันล่าสุด วัน เดือน ปี "
สวัสดีครับ! ยินดีที่ได้ช่วยเหลือคุณในเรื่องข้อมูลสุขภาพของผู้ป่วยนะครับ 😊

{context}

Question: {query}
"""

prompt = ChatPromptTemplate.from_template(template)
qa_chain = prompt | llm | StrOutputParser()


# ---------- Step 6: Helper Function ----------
def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


# ---------- Step 7: Interactive Loop ----------
print("👩‍⚕️ สวัสดีค่ะ พิมพ์คำถามเกี่ยวกับสุขภาพผู้ป่วย หรือพิมพ์ 'exit' เพื่อออก")

while True:
    query = input("❓ คำถามของคุณ: ").strip()
    if query.lower() == "exit":
        print("👋 ขอบคุณที่ใช้บริการนะคะ")
        break

    relevant_docs = retriever.invoke(query)
    answer = qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
    print("💬 คำตอบ:", answer)
