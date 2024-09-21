# Smart-Document-Query

Doc Searcher is a Streamlit application that allows users to query a collection of PDF documents and retrieve precise answers to their queries. This application uses LangChain, HuggingFace, and ChromaDB for document loading, text splitting, embedding, and large language model interactions.

## Features

- Load and process PDF documents.
- Chunk and persist documents for efficient querying.
- Use a large language model to answer questions based on the content of the PDF documents.
- Streamlit-based user interface for easy querying.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/doc-searcher.git
    cd doc-searcher
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a HuggingFace API token and set it as an environment variable:

    ```bash
    export HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token"
    ```

## Usage

1. Place your PDF documents in the specified folder (e.g., `/home/manjeet/Desktop/langchain_tests/consent_forms_cleaned/`).

2. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3. Open your browser and go to the local server address provided by Streamlit (e.g., `http://localhost:8501`).

4. Enter your query in the text input field and press the "Generate" button to get answers based on the content of the PDF documents.

## Project Structure

- `app.py`: Main application file.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
