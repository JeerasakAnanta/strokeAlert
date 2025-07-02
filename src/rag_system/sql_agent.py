import os
import logging
from dotenv import load_dotenv

# langchain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Load environment variables from .env file
load_dotenv(".env")

# Constants for MySQL Database
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_host = os.getenv("MYSQL_HOST")
mysql_port = os.getenv("MYSQL_PORT")
mysql_db = os.getenv("MYSQL_DATABASE")


# ==================== RAG_SQL Agent ====================
def sql_agent(user_input: str) -> str:
    db_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"

    try:
        db = SQLDatabase.from_uri(db_uri)
    except Exception as e:
        return f"[ERROR] Failed to connect to the database: {str(e)}"

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
        return agent_executor.run(user_input)
    except Exception as e:
        return f"[ERROR] {str(e)}"


def assistant_agent(sql_input: str) -> str:
    """
    This function generates a response from the assistant agent
    based on the SQL query results provided as input.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """คุณคือผู้ช่วย อาสาสมัครสาธารณสุขประจำหมู่บ้าน (อสม.) 
        ข้อมูลสุขภาพของผู้ป่วย  คุณสามารถตอบคำถามเกี่ยวกับข้อมูลสุขภาพของผู้ป่วยได้ โดยข้อมูลมี Field อยู่ในฐานข้อมูลมีดังนี้  
        hn               -- หมายเลขผู้ป่วย,
        date             -- วันที่บันทึกข้อมูล
        full_name        -- ชื่อและนามสกุลของผู้ป่วย
        address          -- ที่อยู่ของผู้ป่วย
        bmi              -- ดัชนีมวลกาย (Body Mass Index)
        sbp              -- ความดันโลหิตซิสโตลิก (Systolic Blood Pressure)
        dbp              -- ความดันโลหิตด้านล่าง (Diastolic Blood Pressure)
        blood_sugar      -- ระดับน้ำตาลในเลือด
        bone_status      -- สถานะของกระดูก
        dementia         -- สถานะสมองเสื่อม     (0 = เท็จ, 1 = จริง)
        depression       -- สถานะซึมเศร้า       (0 = เท็จ, 1 = จริง)
        nutrition_status -- สถานะโภชนาการ
        smoking          -- สถานะการสูบบุหรี่     (0 = เท็จ, 1 = จริง).
        recorder         -- ผู้บันทึกข้อมูล    
        ตอบโดยใช้ภาษาไทย และ มี emoji ประกอบการตอบคำถาม
        ตัวอย่างการตอบคำถาม เช่น
        ✅ โปรดตอบโดยใช้ภาษาไทย และใส่ emoji ให้เหมาะสม เช่น
        - 🧑‍⚕️ ชื่อผู้ป่วย
        - 🏠 ที่อยู่
        - ⚖️ BMI
        - 💓 ความดันโลหิต
        - 🍬 น้ำตาลในเลือด
        - 🧠 สมองเสื่อม
        - 😔 ซึมเศร้า
        - 🚬 สูบบุหรี่
        - 📝 บันทึกล่าสุด

        ✅ ตัวอย่างคำตอบที่ดี:
        "🧑‍⚕️นาย ก ขาว  
        🏠 ที่อยู่: 88/8 หมู่ 5 ต.สุขใจ  
        💓 ความดัน: 120/80 mmHg  
        🍬 น้ำตาล: 95 mg/dL  
        ⚖️ BMI: 23.5  
        🧠 ไม่มีภาวะสมองเสื่อม  
        😔 ไม่มีภาวะซึมเศร้า  
        🚬 ไม่สูบบุหรี่  
        📝 บันทึกล่าสุด: 14 กุมภาพันธ์ 2568"
        ยินดีที่ได้ช่วยเหลือคุณในเรื่องข้อมูลสุขภาพของผู้ป่วยนะครับ 😊
        """,
            ),
            ("user", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm
    result_summary = chain.invoke({"input": f"สรุปข้อมูล  {sql_input}"})

    return result_summary.content
