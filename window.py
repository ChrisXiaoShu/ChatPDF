import PySimpleGUI as sg

# Create the layout of the window
layout = [
    [sg.Text("Select a PDF file:")],
    [sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("PDF Files", "*.pdf"),))],
    [sg.Button("Upload"), sg.Button("Cancel")]
]

# Create the window
window = sg.Window("PDF File Uploader", layout)

# Event loop to process window events
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Cancel":
        break
    elif event == "Upload":
        pdf_file = values["-FILE-"]
        # Perform the upload or any further processing here
        print(f"Uploaded file: {pdf_file}")

# Close the window
window.close()