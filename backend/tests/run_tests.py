#!/usr/bin/env python3
"""
Test runner for RAG methods and retrieval strategies
"""
import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print('='*60)
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run RAG methods and retrieval strategies tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--parallel", type=int, help="Number of parallel workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--file", help="Run specific test file")
    parser.add_argument("--function", help="Run specific test function")
    parser.add_argument("--install", action="store_true", help="Install test requirements first")
    
    args = parser.parse_args()
    
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent
    os.chdir(backend_dir)
    
    # Install requirements if requested
    if args.install:
        install_cmd = [sys.executable, "-m", "pip", "install", "-r", "tests/test_requirements.txt"]
        if not run_command(install_cmd, "Installing test requirements"):
            print("Failed to install requirements")
            return 1
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    # Add configuration file
    pytest_cmd.extend(["-c", "tests/pytest.ini"])
    
    # Add specific test selection
    if args.file:
        test_path = f"tests/{args.file}" if not args.file.startswith("tests/") else args.file
        pytest_cmd.append(test_path)
        if args.function:
            pytest_cmd[-1] += f"::{args.function}"
    else:
        pytest_cmd.append("tests/")
    
    # Add marker-based filtering
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.performance:
        markers.append("performance")
    
    if markers:
        pytest_cmd.extend(["-m", " or ".join(markers)])
    
    # Add fast mode (skip slow tests)
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-vv")
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend([
            "--cov=chat_engine",
            "--cov-report=html:reports/htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml:reports/coverage.xml"
        ])
    
    # Add HTML report
    pytest_cmd.extend([
        "--html=reports/pytest_report.html",
        "--self-contained-html"
    ])
    
    # Run tests
    success = run_command(pytest_cmd, "Running RAG tests")
    
    if success:
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("📊 Reports generated in 'reports/' directory:")
        print("   - pytest_report.html: Test execution report")
        if args.coverage:
            print("   - htmlcov/index.html: Coverage report")
            print("   - coverage.xml: Coverage data for CI/CD")
        print("="*60)
        
        # Display quick summary
        if args.coverage:
            coverage_cmd = [sys.executable, "-m", "coverage", "report", "--show-missing"]
            run_command(coverage_cmd, "Coverage Summary")
        
        return 0
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed!")
        print("Check the reports in 'reports/' directory for details")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())