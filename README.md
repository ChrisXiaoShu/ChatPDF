# ChatPDF
## Summary
ChatPDF is a Python project that leverages the LangChain library, a useful vector embedding library, and the OpenAI API to address the issue of ChatGPT's inability to read large PDF files. This project allows users to upload large PDF or text files and ask questions or request a summary of the article using a chatroom-like user interface. The project consists of two key files: window.py, which handles the chatroom user interface, and util.py, which includes utility functions to interact with the LangChain library.

## Project Detail
The ChatPDF project utilizes the LangChain library, specifically the ConversationalRetrievalChain and ChatOpenAI models, to provide a chat-based interface for interacting with large PDF or text files. The project follows a two-step process:

1. Enter your OpenAI API key: Users are required to input their OpenAI API key to authenticate and enable access to the OpenAI language model.
2. Upload a file: Users can upload a PDF or text file, which will be processed and converted into embedding vectors using the LangChain library. These embedding vectors allow the language model to understand and interact with the uploaded content effectively.

Once the file is uploaded, the user can start asking questions or requesting summaries of the article through the chatroom user interface. The input is processed by the ConversationalRetrievalChain model, and the corresponding answer or summary is displayed in the chat history.

The project consists of two main files:

- window.py: This file handles the chatroom user interface using the PySimpleGUI library. It facilitates the input of the OpenAI API key, file upload, and displaying the chat history.
- util.py: This file includes utility functions to interact with the LangChain library. It provides functions for loading PDF or text files, splitting the text into manageable chunks, and creating embedding vector stores using the OpenAIEmbeddings model from LangChain.

## Installation Guide
To set up the ChatPDF project, follow the steps below:
1.  Clone the repository 
```
git clone https://github.com/ChrisXiaoShu/ChatPDF.git
cd ChatPDF
```
2. Create a virtual environment (optional but recommended):
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Run the ChatPDF application:
```
python window.py
```
This will launch the chatroom user interface.

Note: Make sure you have a PDF or text file ready for upload when using the application.

## Usage
Once the application is running, follow these steps to utilize the ChatPDF functionality:

1. Enter your OpenAI API key in the provided text field.
2. Click the "Browse" button and select a PDF or text file from your local machine.
3. Once the file is selected, you can start by click upload button and typing your questions or requests in the input field and press "Send" to get a response.
4. The chat history will display the conversation, including both the user's questions and the AI's answers.
5. You can continue the conversation by entering more questions or requests.

## Acknowledgments
The ChatPDF project utilizes the following libraries and APIs:

- LangChain: A vector embedding library for natural language processing tasks.
- OpenAI API: Provides access to the OpenAI language models for text generation and understanding.
- PySimpleGUI: A Python library for creating simple and interactive graphical user interfaces.
- PyPDF2: A library for reading and manipulating PDF files in Python.