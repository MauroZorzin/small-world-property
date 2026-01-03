"""
Dependency Graph Extraction Module
Extracts dependency graphs from code repositories using the Depends tool
"""

import os
import sys
import subprocess
import logging
import argparse
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from tqdm import tqdm
import yaml


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging with both file and console handlers"""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extraction_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class DependencyExtractor:
    """Handles extraction of dependency graphs using the depends tool"""
    
    def __init__(
        self, 
        depends_path: str, 
        output_folder: Path, 
        language: str, 
        logger: logging.Logger,
        extraction_timeout: int = 300
    ):
        self.depends_path = depends_path
        self.output_folder = Path(output_folder)
        self.language = language
        self.logger = logger
        self.extraction_timeout = extraction_timeout
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_projects": 0,
            "successful": 0,
            "failed": 0,
            "empty": 0,
            "timeout": 0,
            "failed_projects": []
        }
    
    def validate_depends_tool(self) -> bool:
        """Validate that depends executable exists and is accessible"""
        depends_file = Path(self.depends_path)
        if not depends_file.exists():
            self.logger.error(f"Depends executable not found: {self.depends_path}")
            return False
        
        if not depends_file.is_file():
            self.logger.error(f"Depends path is not a file: {self.depends_path}")
            return False
        
        self.logger.info(f"Depends executable validated: {self.depends_path}")
        return True
    
    def extract_project(self, project_folder: Path) -> Optional[Path]:
        """
        Extract dependencies for a single project
        
        Args:
            project_folder: Path to the project source code
            
        Returns:
            Path to generated .dot file if successful, None otherwise
        """
        try:
            project_name = project_folder.name
            output_file = self.output_folder / f"{project_name}.dot"
            
            # Build command
            command = [
                self.depends_path,
                self.language,
                str(project_folder),
                str(output_file),
                "--format=dot",
                "--granularity=file",
                "--external-deps",
                "--output-self-deps",
            ]
            
            self.logger.info(f"Extracting: {project_name}")
            self.logger.debug(f"Command: {' '.join(command)}")
            
            # Run the depends tool
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.extraction_timeout
            )
            
            # Check for errors
            if result.returncode != 0:
                self.logger.error(f"Failed: {project_name}")
                self.logger.error(f"Error: {result.stderr}")
                self.stats["failed"] += 1
                self.stats["failed_projects"].append({
                    "project": project_name,
                    "reason": "non-zero return code",
                    "stderr": result.stderr
                })
                return None
            
            # Validate output file
            if not output_file.exists():
                self.logger.error(f"Output file not created: {project_name}")
                self.stats["failed"] += 1
                self.stats["failed_projects"].append({
                    "project": project_name,
                    "reason": "output file not created"
                })
                return None
            
            if output_file.stat().st_size == 0:
                self.logger.warning(f"Empty output file: {project_name}")
                self.stats["empty"] += 1
                self.stats["failed_projects"].append({
                    "project": project_name,
                    "reason": "empty output file"
                })
                return None
            
            # Success
            file_size = output_file.stat().st_size
            self.logger.info(f"Success: {project_name} ({file_size} bytes)")
            self.stats["successful"] += 1
            return output_file
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout: {project_name} (exceeded {self.extraction_timeout}s)")
            self.stats["timeout"] += 1
            self.stats["failed_projects"].append({
                "project": project_name,
                "reason": "timeout"
            })
            return None
            
        except Exception as e:
            self.logger.error(f"Exception in {project_name}: {e}")
            self.stats["failed"] += 1
            self.stats["failed_projects"].append({
                "project": project_name,
                "reason": f"exception: {str(e)}"
            })
            return None
    
    def extract_all(self, repos_folder: Path) -> List[Path]:
        """
        Extract dependencies for all projects in folder
        
        Args:
            repos_folder: Path to folder containing project subfolders
            
        Returns:
            List of paths to successfully generated .dot files
        """
        # Get all project folders
        if not repos_folder.exists():
            self.logger.error(f"Repositories folder not found: {repos_folder}")
            return []
        
        project_folders = [d for d in repos_folder.iterdir() if d.is_dir()]
        
        if not project_folders:
            self.logger.warning(f"No project folders found in: {repos_folder}")
            return []
        
        self.stats["total_projects"] = len(project_folders)
        self.logger.info(f"Found {len(project_folders)} projects to process")
        
        # Process each project
        extracted_files = []
        for project_folder in tqdm(project_folders, desc="Extracting dependencies"):
            output_file = self.extract_project(project_folder)
            if output_file:
                extracted_files.append(output_file)
        
        return extracted_files
    
    def print_summary(self):
        """Print extraction summary statistics"""
        print("\n" + "=" * 80)
        print("EXTRACTION SUMMARY")
        print("=" * 80)
        print(f"Total projects:      {self.stats['total_projects']}")
        print(f"Successfully extracted: {self.stats['successful']}")
        print(f"Failed:              {self.stats['failed']}")
        print(f"Empty files:         {self.stats['empty']}")
        print(f"Timeouts:            {self.stats['timeout']}")
        print(f"Success rate:        {(self.stats['successful'] / max(self.stats['total_projects'], 1)) * 100:.1f}%")
        print("=" * 80)
        
        if self.stats["failed_projects"]:
            print("\nFailed projects:")
            for failed in self.stats["failed_projects"]:
                print(f"  - {failed['project']}: {failed['reason']}")
    
    def save_report(self, report_file: Path):
        """Save extraction report to JSON file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "depends_path": self.depends_path,
                "output_folder": str(self.output_folder),
                "language": self.language,
                "timeout": self.extraction_timeout
            },
            "statistics": self.stats
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to: {report_file}")


def load_config(config_file: Optional[Path] = None) -> dict:
    """Load configuration from YAML file or use defaults"""
    default_config = {
        "depends_path": "depends-0.9.7-package-20221104a/depends.exe",
        "repos_folder": "repos",
        "output_folder": "depends-out",
        "language": "java",
        "log_folder": "logs",
        "extraction_timeout": 300
    }
    
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Error loading config file: {e}")
            print("Using default configuration")
    
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description="Extract dependency graphs from code repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract with default settings
  python extract_dependencies.py
  
  # Extract with custom configuration
  python extract_dependencies.py --config config.yaml
  
  # Extract specific language
  python extract_dependencies.py --repos ./my-repos --language python
  
  # Extract with custom output folder
  python extract_dependencies.py --output ./my-graphs
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--repos",
        type=Path,
        help="Path to repositories folder"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output folder for .dot files"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["cpp", "python", "java", "kotlin", "go", "ruby", "pom"],
        help="Programming language"
    )
    parser.add_argument(
        "--depends-path",
        type=str,
        help="Path to depends executable"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout per project in seconds (default: 300)"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save extraction report to JSON"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.repos:
        config["repos_folder"] = str(args.repos)
    if args.output:
        config["output_folder"] = str(args.output)
    if args.language:
        config["language"] = args.language
    if args.depends_path:
        config["depends_path"] = args.depends_path
    if args.timeout:
        config["extraction_timeout"] = args.timeout
    
    # Setup paths
    repos_folder = Path(config["repos_folder"])
    output_folder = Path(config["output_folder"])
    log_folder = Path(config["log_folder"])
    
    # Setup logging
    logger = setup_logging(log_folder)
    logger.info("=" * 80)
    logger.info("Dependency Graph Extraction")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Depends path: {config['depends_path']}")
    logger.info(f"  Repos folder: {repos_folder}")
    logger.info(f"  Output folder: {output_folder}")
    logger.info(f"  Language: {config['language']}")
    logger.info(f"  Timeout: {config['extraction_timeout']}s")
    
    # Create extractor
    extractor = DependencyExtractor(
        depends_path=config["depends_path"],
        output_folder=output_folder,
        language=config["language"],
        logger=logger,
        extraction_timeout=config["extraction_timeout"]
    )
    
    # Validate depends tool
    if not extractor.validate_depends_tool():
        logger.error("Depends tool validation failed. Exiting.")
        sys.exit(1)
    
    # Extract dependencies
    logger.info("\nStarting extraction...")
    extracted_files = extractor.extract_all(repos_folder)
    
    # Print summary
    extractor.print_summary()
    
    # Save report if requested
    if args.save_report:
        report_file = output_folder / "extraction_report.json"
        extractor.save_report(report_file)
    
    # Exit with appropriate code
    if extracted_files:
        logger.info(f"\nExtraction completed: {len(extracted_files)} .dot files generated")
        logger.info(f"Output location: {output_folder}")
        sys.exit(0)
    else:
        logger.error("\nNo files were successfully extracted")
        sys.exit(1)


if __name__ == "__main__":
    main()