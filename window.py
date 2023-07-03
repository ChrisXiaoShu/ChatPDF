import PySimpleGUI as sg

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
pdf_file = None

# Event loop to process window events
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Cancel":
        break
    elif event == "Upload":
        pdf_file = values["-FILE-"]
        # Perform the upload or any further processing here
        print(f"Uploaded file: {pdf_file}")
    elif event == "Send":
        # raise alert if no pdf file uploaded
        if pdf_file is None:    
            sg.popup("Please upload a PDF file first!")
            continue
        
        message = values["-INPUT-"]
        chat_history.append(message)
        window["-CHAT_HISTORY-"].update("\n".join(chat_history))
        window["-INPUT-"].update("")

# Close the window
window.close()