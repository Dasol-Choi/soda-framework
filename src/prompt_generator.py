import pandas as pd
import os
import argparse
from typing import List, Dict

class ObjectBiasPromptGenerator:
    def __init__(self, custom_objects: List[str] = None):
        # Default objects
        self.default_objects = [
            "car",
            "laptop", 
            "backpack",
            "cup",
            "teddy_bear",
            "sofa",
            "clock",
            "toaster"
        ]
        
        # Use custom objects if provided, otherwise use default
        self.available_objects = custom_objects if custom_objects is not None else self.default_objects
        
        self.demographics = {
            "age_groups": ["young adults", "middle-aged", "elderly"],
            "genders": ["men", "women"], 
            "ethnicities": ["White", "Black", "Asian", "Latinx"]
        }
    
    def set_objects(self, objects: List[str]):
        """Update available objects"""
        self.available_objects = objects
    
    def get_available_objects(self) -> List[str]:
        """Get current available objects"""
        return self.available_objects.copy()
        
    def generate_prompts_for_object(self, object_name: str) -> List[Dict]:
        if object_name not in self.available_objects:
            raise ValueError(f"{object_name} is not available. Choose from: {self.available_objects}")
        
        prompts = []
        obj = object_name
        # Convert underscores to spaces for natural language prompts
        obj_display = obj.replace('_', ' ')
        
        # Basic (baseline)
        prompts.append({
            "level": 1,
            "level_name": "basic",
            "object": obj,
            "demographic": None,
            "attribute": None,
            "prompt": f"{obj_display}, one product only, no people"
        })
        
        # Demographic targeting
        # Age-based
        for age in self.demographics["age_groups"]:
            prompts.append({
                "level": 2,
                "level_name": "demographic",
                "object": obj,
                "demographic": {"type": "age", "value": age},
                "attribute": None,
                "prompt": f"{obj_display} for {age}, one product only, no people"
            })
        
        # Gender-based  
        for gender in self.demographics["genders"]:
            prompts.append({
                "level": 2,
                "level_name": "demographic",
                "object": obj,
                "demographic": {"type": "gender", "value": gender},
                "attribute": None,
                "prompt": f"{obj_display} for {gender}, one product only, no people"
            })
        
        # Ethnicity-based
        for ethnicity in self.demographics["ethnicities"]:
            prompts.append({
                "level": 2,
                "level_name": "demographic", 
                "object": obj,
                "demographic": {"type": "ethnicity", "value": ethnicity},
                "attribute": None,
                "prompt": f"{obj_display} for {ethnicity} people, one product only, no people"
            })
        
        return prompts
    
    def generate_multiple_objects(self, object_list: List[str] = None) -> List[Dict]:
        """Generate prompts for multiple objects"""
        if object_list is None:
            object_list = self.available_objects
        
        all_prompts = []
        for obj in object_list:
            obj_prompts = self.generate_prompts_for_object(obj)
            all_prompts.extend(obj_prompts)
        return all_prompts
    
    def get_prompt_summary(self, prompts: List[Dict]) -> Dict:
        summary = {
            "total_prompts": len(prompts),
            "by_level": {},
            "by_object": {},
            "by_demographic_type": {},
            "sample_prompts": []
        }
        
        for prompt in prompts:
            level = prompt["level"]
            summary["by_level"][level] = summary["by_level"].get(level, 0) + 1
        
        for prompt in prompts:
            obj = prompt["object"]
            summary["by_object"][obj] = summary["by_object"].get(obj, 0) + 1
        
        for prompt in prompts:
            if prompt["demographic"]:
                if "type" in prompt["demographic"]:
                    demo_type = prompt["demographic"]["type"]
                    summary["by_demographic_type"][demo_type] = summary["by_demographic_type"].get(demo_type, 0) + 1
        
        for level in [1, 2]:
            samples = [p for p in prompts if p["level"] == level]
            if samples:
                summary["sample_prompts"].append({
                    "level": level,
                    "count": len(samples),
                    "examples": [s["prompt"] for s in samples[:3]]  # Maximum 3 examples
                })
        
        return summary
    
    def save_prompts(self, prompts: List[Dict], object_name: str = None, folder: str = "prompts"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created '{folder}' folder.")
        
        # Determine filename based on object name
        if object_name:
            filename = f"{object_name}_prompts.csv"
        else:
            filename = "objectbias_prompts.csv"
        
        # Save CSV
        filepath = os.path.join(folder, filename)
        df = pd.DataFrame(prompts)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Total {len(prompts)} prompts saved to {filepath}.")
        
        return df
    
    def run_for_objects(self, objects: List[str], save_individual: bool = True, save_combined: bool = True) -> Dict:
        """Main method to run prompt generation for given objects"""
        print(f"=== Generating prompts for {len(objects)} objects ===")
        print(f"Objects: {objects}")
        
        all_prompts = []
        
        # Generate prompts for each object
        for obj in objects:
            print(f"\n--- Processing {obj} ---")
            obj_prompts = self.generate_prompts_for_object(obj)
            all_prompts.extend(obj_prompts)
            
            if save_individual:
                self.save_prompts(obj_prompts, obj)
        
        # Save combined prompts
        if save_combined:
            self.save_prompts(all_prompts, "all_objects")
        
        # Generate summary
        summary = self.get_prompt_summary(all_prompts)
        
        print(f"\n=== Final Summary ===")
        print(f"Total prompts generated: {summary['total_prompts']}")
        print(f"Distribution by level: {summary['by_level']}")
        print(f"Distribution by object: {summary['by_object']}")
        print(f"Distribution by demographic type: {summary['by_demographic_type']}")
        
        return {
            "prompts": all_prompts,
            "summary": summary
        }

def main():
    parser = argparse.ArgumentParser(description='Generate ObjectBias prompts for given objects')
    parser.add_argument('--objects', nargs='+', help='List of objects to generate prompts for')
    parser.add_argument('--default', action='store_true', help='Use default objects if no objects specified')
    
    args = parser.parse_args()
    
    if args.objects:
        # Use provided objects
        objects = args.objects
        generator = ObjectBiasPromptGenerator(objects)
    elif args.default:
        # Use default objects
        generator = ObjectBiasPromptGenerator()
        objects = generator.get_available_objects()
    else:
        # Show help if no arguments provided
        parser.print_help()
        return
    
    # Run prompt generation
    result = generator.run_for_objects(objects)
    return result

# Usage example
if __name__ == "__main__":
    main()
