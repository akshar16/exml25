import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def load_single_model(car_folder):
    try:
        model_file = car_folder / "model.py"
        if not model_file.exists():
            return None
        
        print(f"Loading {car_folder.name}...")
        start = time.time()
        
        spec = importlib.util.spec_from_file_location(
            f"{car_folder.name}.model", 
            str(model_file)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        elapsed = time.time() - start
        print(f"Loaded {car_folder.name} in {elapsed:.2f}s")
        
        if hasattr(mod, "model"):
            return (car_folder.name, mod.model)
        else:
            print(f"{car_folder.name} has no 'model' function")
            return None
            
    except Exception as e:
        print(f"Failed to load {car_folder.name}: {e}")
        return None

def load_models_concurrent(models_dir="models", max_workers=4):
    base_path = Path(models_dir)
    
    if not base_path.exists():
        print(f"Models directory '{models_dir}' not found!")
        return []
    
    car_folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    if not car_folders:
        print("No model folders found!")
        return []
    
    print(f"\nLoading {len(car_folders)} models concurrently...")
    start_time = time.time()
    
    models = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {
            executor.submit(load_single_model, folder): folder 
            for folder in car_folders
        }
        
        for future in as_completed(future_to_folder):
            result = future.result()
            if result is not None:
                models.append(result)
    
    total_time = time.time() - start_time
    print(f"\nLoaded {len(models)}/{len(car_folders)} models in {total_time:.2f}s")
    
    return models

if __name__ == "__main__":
    from env.game import F1Game
    game = F1Game()
    models = load_models_concurrent("models", max_workers=4)  
    if not models:
        print("No models loaded! Exiting.")
        exit(1)
    
    print(f"\nStarting game with {len(models)} models...")
    
    control_funcs = [func for name, func in models]
    game.run(control_funcs)