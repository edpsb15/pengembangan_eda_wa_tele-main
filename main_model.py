from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.cache import InMemoryCache
#from langchain.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
from API_GEMINI import GOOGLE_API_KEY
import re
import uuid

app = Flask(__name__)

# Load FAISS index
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Set up the cache (using InMemoryCache)
set_llm_cache(InMemoryCache()) 

# Store session data
store = {}

def remove_emojis(text):
    """Remove emojis from text."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def get_conversational_chain():
    """Create and return a QA chain with the provided context."""
    prompt_template = """
    Anda adalah EDA (Electronic Data Assistance) pada aplikasi WhatsApp yang membantu pengguna berkonsultasi dengan pertanyaan statistik dan melayani permintaan data khususnya dari BPS Kabupaten Samosir. Sebagai kaki tangan BPS Kabupaten Samosir, Anda tidak boleh mendiskreditkan BPS Kabupaten Samosir. Anda juga meyakinkan pengguna bahwa data yang Anda peroleh benar adanya.
    
    Informasi yang perlu Anda ketahui jika ada pengguna yang bertanya adalah Kepala BPS Kabupaten Samosir adalah Devitanorani Saragih, SST, M.Stat. Kantor BPS Kabupaten Samosir berlokasi di Komplek Perkantoran Pemkab Samosir, Parbaba, Siopat Sosor, Pangururan, Samosir, Sumatera Utara 22932. Visi BPS pada tahun 2025 adalah menjadi penyedia data statistik berkualitas untuk Indonesia Maju. Misi BPS pada tahun 2024 meliputi: 1) Menyediakan statistik berkualitas yang berstandar
    nasional dan internasional; 2) Membina K/L/D/I melalui Sistem Statistik Nasional yang berkesinambungan; 3) Mewujudkan pelayanan prima di bidang statistik untuk terwujudnya Sistem Statistik Nasional; 4) Membangun SDM yang unggul dan adaptif berlandaskan nilai profesionalisme, integritas, dan amanah.

    Hanya dalam percakapan sekali dan pertama kali, Anda akan memberikan penafian bahwa pesan Anda terkirim dalam waktu 10 hingga 20 detik, riwayat chat akan terhapus tiap jam, melarang kata-kata berbau SARA, dan menghimbau pengguna untuk menggunakan kalimat yang lengkap dilengkapi wilayah dan tahun data serta tidak menggunakan singkatan atau akronim untuk data yang akurat. Anda juga dapat bertanya mengenai nama dan umur pengguna, dan berbicara sesuai dengan umur pengguna. Jika pengguna berumur lebih dari 30 tahun, Anda memanggil Pak/Bu.

    Anda tidak menerima input berupa audio dan gambar. Anda menerima input penerimaan data dari pengguna dengan format wilayah dan tahun saja. Jika ada pengguna meminta data diluar format, Anda memberikan saran format yang benar. Output Anda dapat berupa teks atau tabel.

    Anda berikan jawaban yang relevan dan ringkas berdasarkan dokumen di bawah ini dan pertanyaan dari pengguna. Anda juga tidak memberikan contoh data di luar dokumen. Jika ada permintaan data di luar dokumen, arahkan pengguna ke https://samosirkab.bps.go.id atau Pelayanan Statistik Terpadu (PST) di BPS Kabupaten Samosir untuk informasi lebih lanjut. Anda memberikan alasan ketidatersediaan data karena sistem Anda masih dalam tahap pengembangan lalu arahkan pengguna ke https://samosirkab.bps.go.id atau Pelayanan Statistik Terpadu (PST) di BPS Kabupaten Samosir untuk informasi lebih lanjut. Jika ada yang bisa dihubungi, Anda menyarankan mengunjungi kantor atau PST melalui pst1217@bps.go.id.
    
    Konteks:\n {context}\n
    Pertanyaan Pengguna: \n{input}\n    
    Jawaban yang relevan (berdasarkan dokumen):\n
    """

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    # retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2})
    # retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_tokens=None, google_api_key=GOOGLE_API_KEY)

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_response(user_question, session_id):
    """Get response from the model using RAG."""
    try:
        # Perform similarity search using FAISS
        # docs = vector_store.similarity_search(user_question)
        
        # Prepare context from the retrieved documents
        # context = "\n".join([doc.page_content for doc in docs])

        # Get the QA chain
        rag_chain = get_conversational_chain()

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Set configuration with the given session_id
        config = {"configurable": {"session_id": session_id}}

        # Invoke the chain and get the response
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config=config
        )

        # Extract the answer text
        response_text = response.get("answer", "")

        # Check for specific response codes
        response_code = response.get("response_code", 200)
        if response_code == 429:
            return "Maaf, layanan saat ini sedang sibuk. Silakan coba lagi nanti."
        elif response_code == 503:
            return "Layanan saat ini tidak tersedia. Silakan coba lagi nanti."

        # Remove emojis from the response
        response_text = remove_emojis(response_text)

        # Include context in the response
        # if context:
        #     response_text = f"Context retrieved:\n{context}\n\nAnswer:\n{response_text}"

    except Exception as e:
        response_text = f"Error: {str(e)}"

    return response_text
    # return "Data sedang diperbaiki. Coba lagi nanti"

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    response_text = data.get('response_text')
    notelp = data.get('notelp')  # Get the notelp from the request, if provided
    
    if response_text:
        # If notelp is not provided, generate a new one
        if not notelp:
            notelp = str(uuid.uuid4())

        # Process the input text and get the response
        processed_text = get_response(response_text, notelp)

        return jsonify({"status": "success", "processed_text": processed_text, "notelp": notelp}), 200
    
    return jsonify({"status": "error", "message": "No response text provided"}), 400

if __name__ == '__main__':
    app.run(port=5002)  # Change port if needed
