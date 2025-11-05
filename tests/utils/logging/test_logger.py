"""
Logging utility for test results and errors
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import json
import traceback
from typing import Any, Dict, List, Optional
import pytest

@pytest.mark.no_collect  # Exclude from test collection
class TestLogger:
    def __init__(self, test_name: str):
        """
        Initialize the test logger.
        
        Args:
            test_name: Name of the test being run
        """
        self.test_name = test_name
        self.test_output_dir = Path(__file__).parent.parent.parent / "test_output"
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Create timestamped files for this test run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.test_output_dir / f"{test_name}_results_{timestamp}.json"
        self.error_file = self.test_output_dir / f"{test_name}_errors_{timestamp}.log"
        
        # Initialize results dictionary
        self.results = {
            "test_name": test_name,
            "timestamp": timestamp,
            "platform": sys.platform,
            "python_version": sys.version,
            "steps": [],
            "summary": {
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "start_time": datetime.now().isoformat()
            }
        }

    def start_test(self, test_description: str):
        """Start a new test with a description."""
        self.log_step("Test Start", True, {"description": test_description})

    def end_test(self, success: bool):
        """End the current test and record its final status."""
        self.log_step("Test End", success, {"final_status": "success" if success else "failure"})
        self.finalize()

    def log_info(self, message: str):
        """Log an informational message."""
        self.log_step("Info", True, {"message": message})

    def log_success(self, message: str):
        """Log a success message."""
        self.log_step("Success", True, {"message": message})

    def log_warning(self, message: str):
        """Log a warning message."""
        self.log_step("Warning", True, {"message": message})

    def log_error(self, message: str, error: Optional[Exception] = None):
        """Log an error message."""
        self.log_step("Error", False, {"message": message}, error)

    def log_step(self, step_name: str, status: bool, details: Dict[str, Any], error: Optional[Exception] = None):
        """
        Log a test step result.
        
        Args:
            step_name: Name of the test step
            status: True if step succeeded, False if failed
            details: Dictionary containing step details
            error: Optional exception if step failed
        """
        step_info = {
            "name": step_name,
            "status": "success" if status else "failure",
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        if error:
            step_info["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
            # Also write to error log
            self._log_error(step_name, error)
        
        self.results["steps"].append(step_info)
        self.results["summary"]["total_steps"] += 1
        if status:
            self.results["summary"]["successful_steps"] += 1
        else:
            self.results["summary"]["failed_steps"] += 1
        
        # Write updated results to file
        self._save_results()

    def _log_error(self, context: str, error: Exception):
        """
        Write error details to error log file.
        
        Args:
            context: Context where error occurred
            error: The exception to log
        """
        timestamp = datetime.now().isoformat()
        with open(self.error_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Error Type: {type(error).__name__}\n")
            f.write(f"Error Message: {str(error)}\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
            f.write(f"{'='*50}\n")

    def _save_results(self):
        """Save current results to JSON file."""
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    def finalize(self):
        """
        Finalize test results and save summary.
        """
        self.results["summary"]["end_time"] = datetime.now().isoformat()
        self._save_results()
        
        # Print summary
        print(f"\nTest Results Summary for {self.test_name}:")
        print(f"Total Steps: {self.results['summary']['total_steps']}")
        print(f"Successful: {self.results['summary']['successful_steps']}")
        print(f"Failed: {self.results['summary']['failed_steps']}")
        print(f"\nResults saved to: {self.results_file}")
        if os.path.exists(self.error_file) and os.path.getsize(self.error_file) > 0:
            print(f"Errors logged to: {self.error_file}")