import typer

from xtrain.run.train import train

app = typer.Typer()

app.command(help="Train a model")(train)

@app.command()
def version():
    typer.echo("xtrain v0.0.1")

if __name__ == "__main__":
    app()
