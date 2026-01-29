import gradio as gr
from src.Pipeline.predict_piepline import CustomData, PredictPipeline

def predict_math_score(
    gender,
    race_ethnicity,
    parental_level_of_education,
    lunch,
    test_preparation_course,
    reading_score,
    writing_score
):
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=float(reading_score),
        writing_score=float(writing_score)
    )

    pred_df = data.get_data_as_data_frame()

    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(pred_df)

    return f"Predicted Math Score: {round(result[0], 2)}"


interface = gr.Interface(
    fn=predict_math_score,
    inputs=[
        gr.Dropdown(["male", "female"], label="Gender"),
        gr.Dropdown(
            ["group A", "group B", "group C", "group D", "group E"],
            label="Race/Ethnicity"
        ),
        gr.Dropdown(
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ],
            label="Parental Education"
        ),
        gr.Dropdown(["standard", "free/reduced"], label="Lunch"),
        gr.Dropdown(["none", "completed"], label="Test Preparation Course"),
        gr.Slider(0, 100, label="Reading Score"),
        gr.Slider(0, 100, label="Writing Score"),
    ],
    outputs="text",
    title="Student Exam Performance Predictor",
    description="Predict Math score using ML pipeline"
)

interface.launch()
