# 📊 AI-Powered Data Visualization Dashboard

An interactive web application built with Streamlit that allows users to upload CSV files, create beautiful visualizations, and get AI-powered insights using the Groq API with LLaMA 3.3 model.

## 🌟 Features

- **CSV File Upload**: Easy drag-and-drop or file selection interface
- **Data Preview**: View and explore your data with summary statistics
- **Interactive Visualizations**: Create various chart types including:
  - Line Charts
  - Bar Charts
  - Scatter Plots
  - Histograms
  - Box Plots
  - Pie Charts
- **AI-Powered Insights**: Generate intelligent analysis of your data using Groq's LLaMA 3.3 model
- **Chat with Your Data**: Ask questions about your dataset and get AI-powered answers
- **Modern UI**: Clean, intuitive interface with responsive design

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualization library
- **Groq API**: AI reasoning with LLaMA 3.3-70b-versatile model

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API Key (get one from [Groq Console](https://console.groq.com))

## 🚀 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/varun/web2/ai-dashboard
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Groq API key:**
   
   You can set the API key in several ways:
   
   **Option 1: Environment variable (Recommended)**
   ```bash
   export GROQ_API_KEY='your_api_key_here'
   ```
   
   **Option 2: Add to your shell profile**
   ```bash
   echo 'export GROQ_API_KEY="your_api_key_here"' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   **Option 3: Create a .env file**
   ```bash
   echo 'GROQ_API_KEY=your_api_key_here' > .env
   ```

## 🎯 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   The app will automatically open in your default browser at `http://localhost:8501`

3. **Upload your data:**
   - Click the file uploader in the sidebar
   - Select a CSV file from your computer
   - The data will be loaded and ready for analysis

4. **Explore your data:**
   - **Data Preview**: View sample data, statistics, and column information
   - **Visualizations**: Select chart types and columns to create interactive plots
   - **AI Insights**: Click "Generate AI Insights" to get intelligent analysis
   - **Chat with Data**: Ask questions about your dataset in natural language

## 📁 Project Structure

```
/project
├── app.py              # Main Streamlit application
├── ai.py               # Groq API integration and helper functions
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🔑 Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and set it as an environment variable

## 📊 Example Use Cases

- **Sales Analysis**: Upload sales data to identify trends and patterns
- **Customer Insights**: Analyze customer behavior and demographics
- **Financial Data**: Visualize financial metrics and get AI recommendations
- **Scientific Research**: Explore experimental data and statistical relationships
- **Business Intelligence**: Transform raw data into actionable insights

## 🎨 Visualization Types

### Line Chart
Perfect for showing trends over time or continuous data.

### Bar Chart
Ideal for comparing values across different categories.

### Scatter Plot
Great for identifying relationships between two numeric variables.

### Histogram
Shows the distribution of a single numeric variable.

### Box Plot
Displays statistical distribution and identifies outliers.

### Pie Chart
Visualizes proportions and percentages of categorical data.

## 💡 Tips for Best Results

1. **Data Format**: Ensure your CSV has a header row with column names
2. **Numeric Data**: Most visualizations work best with numeric columns
3. **File Size**: For large datasets, consider sampling for faster processing
4. **Questions**: Ask specific questions in the chat feature for better AI responses
5. **Insights**: Generate AI insights after exploring your visualizations

## 🤖 AI Features

### Generate Insights
The AI analyzes your entire dataset and provides:
- Key observations about the data
- Interesting patterns and trends
- Potential correlations between variables
- Suggestions for further analysis
- Data quality observations

### Chat with Data
Ask natural language questions like:
- "What are the main trends in this data?"
- "Which columns are most correlated?"
- "What's the average value of [column]?"
- "Are there any outliers in [column]?"
- "What recommendations do you have for this data?"

## 🔧 Troubleshooting

### API Key Issues
If you get an error about the API key:
```bash
# Verify your API key is set
echo $GROQ_API_KEY

# If empty, set it again
export GROQ_API_KEY='your_api_key_here'
```

### Import Errors
If you get module import errors:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### CSV Loading Issues
- Ensure your CSV is properly formatted
- Check for special characters in column names
- Verify the file encoding (UTF-8 recommended)

## 📝 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review the Groq API documentation
- Consult the Streamlit documentation

## 🔄 Version History

- **v1.0.0** - Initial release with core features
  - CSV upload and preview
  - Multiple visualization types
  - AI insights generation
  - Chat with data feature

---

**Built with ❤️ using Streamlit, Plotly, and Groq AI**


