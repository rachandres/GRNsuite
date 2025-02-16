import click
from grnsuite import preprocessing, spike_detection

@click.group()
def cli():
    """GRNsuite - Analyze insect taste electrophysiology recordings"""
    pass

@click.command()
@click.argument("input_file")
@click.option("--output", default="output.csv", help="Output file name")
def analyze(input_file, output):
    """Run the full analysis pipeline"""
    print(f"Analyzing {input_file}...")  # Placeholder
    # Later, load data -> preprocess -> detect spikes -> analyze
    print(f"Results saved to {output}")

cli.add_command(analyze)

if __name__ == "__main__":
    cli()
