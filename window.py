import PySimpleGUI as sg
from lib.util import create_embedding_vectorstore, load_file, split_text
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Create the layout of the window
layout = [
    [sg.Text("Select a PDF file:")],
    [sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("PDF Files", "*.pdf"),))],
    [sg.Button("Upload"), sg.Button("Cancel")],
    [sg.Text("Chat History", font=("Arial", 12), key="-CHAT_HISTORY-", size=(60, 10), justification="left", auto_size_text=True, relief=sg.RELIEF_SUNKEN)],
    [sg.InputText(key="-INPUT-", size=(40, 1)), sg.Button("Send")]
]

# Create the window
window = sg.Window("PDF File Uploader", layout)

# Chat history list
chat_history = []
ui_history = []
pdf_file = None
qa = None

# Event loop to process window events
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Cancel":
        break
    elif event == "Upload":
        pdf_file = values["-FILE-"]
        # load file
        loader = load_file(pdf_file)
        # split text
        texts = split_text(loader)
        # create embedding vectorstore
        vectorstore = create_embedding_vectorstore(texts)
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())
        
    elif event == "Send":
        # raise alert if no pdf file uploaded
        if pdf_file is None or qa is None:    
            sg.popup("Please upload a PDF file first!")
            continue
        
        question = values["-INPUT-"]
    
        answer = qa({"question": question, 'chat_history': chat_history})
        
        ui_history.append("Q: " + question)
        ui_history.append("A: " + answer['answer'])
        
        chat_history.append((question, answer['answer']))
        
        window["-CHAT_HISTORY-"].update("\n".join(ui_history))
        window["-INPUT-"].update("")

# Close the window
window.close()