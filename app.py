import torch
import whisper
from transformers import pipeline
import gradio as gr
import concurrent.futures

# ✅ Load models once (Prevents reloading in every function call)
print("Loading models...")
whisper_model = whisper.load_model("small")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_generator = pipeline("text2text-generation", model="google/flan-t5-large")
print("Models loaded successfully!")

def transcribe_audio(audio_path):
    print("Transcribing audio...")
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    print("Summarizing text using BART...")
    text_chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]  
    summaries = summarizer(text_chunks, max_length=200, min_length=50, do_sample=False)
    return " ".join([s['summary_text'] for s in summaries])

def generate_questions(text):
    print("Generating questions using FLAN-T5...")
    text_chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    questions = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_questions = [
            executor.submit(
                lambda chunk: question_generator(
                    f"You are an AI tutor. Your task is to generate **insightful, topic-specific** questions based on the following passage. Ensure that the questions are relevant to the **key concepts, definitions, and explanations** present in the text. Avoid generic questions.\n\nPassage:\n{chunk}", 
                    max_length=120, num_return_sequences=3, do_sample=True
                ),
                chunk
            ) for chunk in text_chunks
        ]
        
        for future in future_questions:
            generated = future.result()
            questions.extend([q['generated_text'] for q in generated])
    
    return "\n".join(questions)

def process_audio(audio_path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        transcribe_future = executor.submit(transcribe_audio, audio_path)
        transcript = transcribe_future.result()
        
        summarize_future = executor.submit(summarize_text, transcript)
        questions_future = executor.submit(generate_questions, transcript)
        
        summary = summarize_future.result()
        questions = questions_future.result()
    
    combined_text = f"📝 Transcription:\n{transcript}\n\n📜 Summary:\n{summary}\n\n🤔 Practice Questions:\n{questions}"
    file_path = "lecture_summary.txt"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    return transcript, summary, questions, file_path

def gradio_interface(audio):
    return process_audio(audio)

with gr.Blocks(css="""
    #submit-btn, #download-btn {
        background-color: blue !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
    }
    textarea {
        border: 2px solid black !important;
        border-radius: 5px !important;
    }
""") as demo:
    gr.Markdown("# 🎙 LectureGenie: Transcribe, Summarize & Quiz")
    gr.Markdown("Upload a lecture audio file. The system will **transcribe**, **summarize**, and **generate questions** automatically.")

    audio_input = gr.Audio(type="filepath", label="🎤 Upload Audio File", interactive=True)
    submit_btn = gr.Button("Submit", elem_id="submit-btn")

    with gr.Row():
        with gr.Column():
            transcript_box = gr.Textbox(label="📝 Transcription", lines=10, interactive=False, show_copy_button=True)

        with gr.Column():
            summary_box = gr.Textbox(label="📜 Summary", lines=10, interactive=False, show_copy_button=True)

        with gr.Column():
            questions_box = gr.Textbox(label="🤔 Practice Questions", lines=10, interactive=False, show_copy_button=True)

    download_btn = gr.File(label="📥 Download All", interactive=False, visible=False)
    download_button = gr.Button("📥 Download", elem_id="download-btn")

    submit_btn.click(
        gradio_interface,
        inputs=[audio_input],
        outputs=[transcript_box, summary_box, questions_box, download_btn]
    )

    download_button.click(lambda x: x, inputs=[download_btn], outputs=[download_btn])

demo.launch(share=True)
