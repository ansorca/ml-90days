import typer
from stage1.bootstrap import create_classifier
import numpy as np


def validate_features(value: list[str]) -> list[list[float]]:
    result = []
    for sample in value:
        sample_values = sample.split(",")
        if len(sample_values) != 4:
            raise typer.BadParameter(f"Expected 4 features, got {len(sample_values)}")
        result.append([float(v) for v in sample_values])
    return result

app = typer.Typer()

@app.command()
def predict(classifier: str = typer.Option("hybrid", help="Classifier to use"), 
            features: list[str] = typer.Option(..., help="Features values", callback=validate_features)):
    (classifier, class_names) = create_classifier(classifier)
    predictions = classifier.predict( np.array(features, dtype=np.float32))
    for name in  [class_names[i] for i in predictions]:
        typer.echo(name)

if __name__ == "__main__":
    app()