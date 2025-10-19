from ultralytics import YOLO
import torch
import os
from rich.console import Console
from rich.table import Table

console = Console()

def train(user_input, dataset_yaml, device):
    if user_input.lower().strip() == "no":
        console.print("[bold cyan]🚀 Bắt đầu train mới...[/bold cyan]")
        # Load a model
        model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML
        model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data=dataset_yaml, epochs=200, patience=50, imgsz=640, device=device)
    else:
        console.print("[bold green]👉 Tiếp tục train từ model cũ...[/bold green]")
        model = YOLO(r"model\best.pt")

        results = model.train(data=dataset_yaml, epochs=100, imgsz=640, device=device)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold yellow]Thiết bị sử dụng:[/bold yellow] {device}\n")

    path_dir = os.getcwd()
    dataset_dir = os.path.join(path_dir, "datasets")
    list_datasets = {}

    # Hiển thị bảng dataset
    table = Table(title="Danh sách dataset khả dụng", show_lines=True)
    table.add_column("STT", justify="center", style="cyan", no_wrap=True)
    table.add_column("Tên thư mục", justify="left", style="bold white")
    table.add_column("Đường dẫn data.yaml", justify="left", style="green")

    for idx, dirname in enumerate(os.listdir(dataset_dir)):
        dataset_yaml = os.path.join(dataset_dir, dirname, "data.yaml")

        list_datasets[str(idx)] = dataset_yaml
        table.add_row(str(idx), dirname, dataset_yaml)
    
    console.print(table)
    continueOrNew = console.input("[bold yellow]Continue or New train? (yes (skipped)/no)[/bold yellow]: ")
    console.print()
    choiceDataset = console.input("[bold cyan]Chọn dataset theo STT: [/bold cyan]")

    dataset_yaml_choice = list_datasets.get(choiceDataset.strip())

    if dataset_yaml_choice is None:
        console.print("[bold red]❌ Lựa chọn không hợp lệ![/bold red]")

    console.print(f"\n[bold green]✅ Dataset được chọn:[/bold green] {dataset_yaml_choice}")
    train(continueOrNew, dataset_yaml_choice, device)