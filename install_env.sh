
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

sudo apt-get update
sudo apt-get install zstd
curl -fsSL https://ollama.com/install.sh | sh

ollama serve &

ollama pull smollm2:1.7b
ollama pull all-minilm:latest

uv sync --group api
