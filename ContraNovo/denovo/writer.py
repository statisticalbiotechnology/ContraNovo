import csv
import json


class MZTabWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.rows = []
        self.metadata = {}  # Store metadata as a dictionary

    def set_metadata(self, **kwargs):
        # Update metadata with provided key-value pairs
        self.metadata.update(kwargs)

    def append(self, data):
        # Transform the input data into a format suitable for MZTab.
        self.rows.append(data)

    def save(self):
        # Write metadata and data to the file
        with open(self.file_path, "w", newline="") as f:
            # First, write the metadata section.
            for key, value in self.metadata.items():
                f.write(f"# {key}: {value}\n")

            # Leave a blank line between metadata and data (optional)
            f.write("\n")

            # Define the headers for your MZTab file.
            headers = [
                "spectrum_index",
                "precursor_charge",
                "precursor_mz",
                "spectrum_embedding",
            ]

            # Initialize and write data using the csv.DictWriter
            writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
            writer.writeheader()
            for row in self.rows:
                row["spectrum_embedding"] = json.dumps(row["spectrum_embedding"])
                writer.writerow(row)

        print(f"Results saved to {self.file_path}")
