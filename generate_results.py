#!/usr/bin/env python3
"""
Generate results README from JSON files in the results folder.
Creates a markdown table summarizing all results.
"""

import os
import json
import glob
from typing import Dict, List, Tuple


# Configuration for which columns to display for each assignment
ASSIGNMENT_COLUMNS = {
    "cv01": ["NExp", "BestAcc", "EXPS"],
}


def load_json_results(results_dir: str = "results") -> Dict[str, Dict]:
    """
    Load all JSON files from the results directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Dictionary mapping filename to parsed JSON content
    """
    results = {}
    json_pattern = os.path.join(results_dir, "*.json")
    
    for json_file in glob.glob(json_pattern):
        filename = os.path.basename(json_file)
        assignment_name = filename.replace('.json', '')
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results[assignment_name] = data
                print(f"[INFO] Loaded {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            
    return results


def parse_json_structure(data: Dict) -> List[Tuple[str, str, int]]:
    """
    Parse the JSON structure and extract key-value pairs with original sort order.
    
    Args:
        data: JSON data dictionary
        
    Returns:
        List of (metric_name, metric_value, sort_key) tuples
    """
    parsed_data = []
    
    for key, value in data.items():
        # Try to convert key to int for proper sorting
        try:
            sort_key = int(key)
        except ValueError:
            sort_key = 999  # Put non-numeric keys at the end
            
        if isinstance(value, list) and len(value) == 2:
            # Format: {"0": ["DATA_HIST", "-"]}
            metric_name = value[0]
            metric_value = str(value[1])
        else:
            # Fallback for other formats
            metric_name = str(key)
            metric_value = str(value)
            
        parsed_data.append((metric_name, metric_value, sort_key))
    
    return parsed_data


def generate_markdown_table(results: Dict[str, Dict]) -> str:
    """
    Generate a markdown table from the results data.
    Assignments as rows, metrics as columns.
    Only displays configured columns for each assignment.
    
    Args:
        results: Dictionary of assignment results
        
    Returns:
        Markdown formatted table as string
    """
    if not results:
        return "No results found.\n"
    
    markdown = "# Results Summary\n\n"
    
    # Collect metrics for each assignment with their sort keys
    all_metrics = {}  # metric_name -> sort_key
    assignment_data = {}
    
    for assignment_name, data in sorted(results.items()):
        parsed_data = parse_json_structure(data)
        # Create dict with metric_name -> value, and collect sort keys
        assignment_dict = {}
        for metric_name, metric_value, sort_key in parsed_data:
            assignment_dict[metric_name] = metric_value
            all_metrics[metric_name] = sort_key
            
        assignment_data[assignment_name] = assignment_dict
    
    # Determine which metrics to display
    displayed_metrics = set()
    for assignment_name in assignment_data.keys():
        # Get configured columns for this assignment, or all if not configured
        if assignment_name in ASSIGNMENT_COLUMNS:
            configured_columns = ASSIGNMENT_COLUMNS[assignment_name]
            # Only add columns that actually exist in the data
            for col in configured_columns:
                if col in assignment_data[assignment_name]:
                    displayed_metrics.add(col)
        else:
            # If no configuration, show all columns for this assignment
            displayed_metrics.update(assignment_data[assignment_name].keys())
    
    # Sort displayed metrics by their original dictionary keys
    sorted_metrics = sorted(displayed_metrics, key=lambda x: all_metrics.get(x, 999))
    
    if sorted_metrics and assignment_data:
        # Create table header
        markdown += "| Assignment |"
        for metric in sorted_metrics:
            escaped_metric = metric.replace('|', '\\|')
            markdown += f" {escaped_metric} |"
        markdown += "\n"
        
        # Create separator row
        markdown += "|------------|"
        for _ in sorted_metrics:
            markdown += "-----------|"
        markdown += "\n"
        
        # Add data rows
        for assignment_name in sorted(assignment_data.keys()):
            escaped_assignment = assignment_name.upper().replace('|', '\\|')
            markdown += f"| {escaped_assignment} |"
            
            for metric in sorted_metrics:
                value = assignment_data[assignment_name].get(metric, "-")
                escaped_value = str(value).replace('|', '\\|')
                markdown += f" {escaped_value} |"
            markdown += "\n"
        
        markdown += "\n"
    else:
        markdown += "No data available.\n\n"
    
    return markdown


def generate_summary_stats(results: Dict[str, Dict]) -> str:
    """
    Generate summary statistics from all results.
    
    Args:
        results: Dictionary of assignment results
        
    Returns:
        Markdown formatted summary
    """
    summary = "## Summary Statistics\n\n"
    
    total_assignments = len(results)
    summary += f"- **Total Assignments**: {total_assignments}\n"
    
    # Count metrics across all assignments
    all_metrics = set()
    best_acc_values = []
    
    for assignment_name, data in results.items():
        parsed_data = parse_json_structure(data)
        for metric_name, metric_value, sort_key in parsed_data:
            all_metrics.add(metric_name)
            
            # Collect BestAcc values for analysis
            if metric_name == "BestAcc" and metric_value != "-":
                try:
                    acc_value = float(metric_value)
                    best_acc_values.append(acc_value)
                except ValueError:
                    pass
    
    summary += f"- **Total Metrics**: {len(all_metrics)}\n"
    
    if best_acc_values:
        max_acc = max(best_acc_values)
        min_acc = min(best_acc_values)
        avg_acc = sum(best_acc_values) / len(best_acc_values)
        summary += f"- **Best Accuracy Range**: {min_acc:.2f}% - {max_acc:.2f}%\n"
        summary += f"- **Average Best Accuracy**: {avg_acc:.2f}%\n"
    
    summary += "\n"
    return summary


def main():
    """Main function to generate results README."""
    results_dir = "results"
    output_file = os.path.join(results_dir, "README.md")
    
    print(f"[INFO] Loading JSON results from {results_dir}")
    results = load_json_results(results_dir)
    
    if not results:
        print("[WARNING] No JSON files found in results directory")
        return
    
    print(f"[INFO] Generating markdown for {len(results)} assignments")
    
    # Generate markdown content
    markdown_content = generate_markdown_table(results)
    markdown_content += generate_summary_stats(results)
    
    # Add generation timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content += f"\n---\n*Generated on {timestamp}*\n"
    
    # Write to README.md
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"[SUCCESS] Results written to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write {output_file}: {e}")
    
    # Also print to console
    print("\n" + "="*50)
    print("GENERATED README CONTENT:")
    print("="*50)
    print(markdown_content)

