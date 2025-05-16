from transformers import pipeline
import gradio as gr

# Load the summarization model
print("🔄 Loading model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("✅ Model loaded.")

# Define the summarization function
def summarize_text(text):
    print("📝 Received input:", text)

    # Check if the input is empty
    if not text.strip():
        return "⚠️ Please enter some text to summarize."

    try:
        # Generate the summary
        summary = summarizer(text, max_length=60, min_length=30, do_sample=False)
        print("✅ Summary generated:", summary[0]['summary_text'])
        return summary[0]['summary_text']
    except Exception as e:
        print("❌ Error during summarization:", str(e))
        return f"❌ An error occurred: {str(e)}"

# Create the Gradio interface
app = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, placeholder="Paste a few paragraphs of text here...", label="Input Text"),
    outputs=gr.Textbox(label="Summary"),
    title="📝 Text Summarizer",
    description="This app summarizes long text using Hugging Face's BART model."
)

# Launch the app
app.launch()
