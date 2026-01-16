
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
