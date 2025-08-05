Crebrum: Your Second Brain
Crebrum is a powerful, local-first, and privacy-focused "second brain" application designed to help you capture, connect, and retrieve your thoughts and knowledge effortlessly. It leverages the power of AI to provide semantic search and other smart features, running entirely on your local machine.

🌟 Key Features
🧠 Local-First & Private: All your data is stored locally on your machine. No cloud, no servers, no tracking. You have complete ownership and control of your knowledge.

✍️ WYSIWYG Markdown Editor: A beautiful and intuitive editor for formatting your notes with Markdown.

🔗 Bidirectional Linking: Create a network of your knowledge by linking notes together. See all the connections to a specific note automatically.

🕸️ Graph View: Visualize the connections between your notes as a graph, helping you discover new relationships and insights.

🤖 AI-Powered Semantic Search: Find what you're looking for based on meaning and context, not just keywords. Crebrum uses local AI models to understand your queries.

💬 Chat with your Brain: Interact with your notes through a conversational interface. Ask questions and get answers from your own knowledge base.

🖼️ Multi-Modal: Store and search through various types of media, including text, images, and more.

🔌 Extensible: Built with extensibility in mind to allow for future plugins and integrations.

💻 Cross-Platform: Available for macOS, Windows, and Linux.

<img width="293" height="218" alt="image" src="https://github.com/user-attachments/assets/3057c34e-210c-4ecd-9d9e-c498071a6853" />



🚀 Getting Started
Prerequisites
Ensure you have git and npm (or yarn/pnpm) installed on your system.

Installation
Clone the repository:

git clone https://github.com/forUAi/crebrum.git

Navigate to the project directory:

cd crebrum

Install dependencies:

npm install
# or
yarn install
# or
pnpm install

Running the Application
To start the application in development mode, run:

npm run dev

🛠️ How it Works
Crebrum is built using a modern tech stack to provide a robust and efficient experience:

Tauri: A framework for building lightweight, secure, and fast cross-platform desktop apps using web technologies.

React: A JavaScript library for building user interfaces.

Rust: Powers the backend logic, providing performance and safety for file operations and AI computations.

SQLite: Used for robust, local data storage.

LanceDB: An embedded vector database for high-performance semantic search.

The application runs a local server and uses vector embeddings to create a searchable index of your notes. This allows the AI to understand the semantic relationships between your ideas and provide intelligent search results.

🤝 Contributing
Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, your help is appreciated. Please feel free to open an issue or submit a pull request.

Fork the repository.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
