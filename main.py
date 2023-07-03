from lib.util import load_file, split_text, create_embedding_vectorstore
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

def main():
    # load file from user input
    file_path = input("Enter file path: ")
    loader = load_file(file_path)

    # split text
    texts = split_text(loader)

    # create embedding vectorstore
    vectorstore = create_embedding_vectorstore(texts)
    
    # start qa conversation
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())
    chat_history = []
    while True:
        question = input("Enter question: ")
        answer = qa({"question": question, 'chat_history': chat_history})
        print('answer:', answer['answer'])
        chat_history.append((question, answer['answer']))
    
if __name__ == "__main__":
    main()