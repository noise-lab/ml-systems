# ML Systems Midterm 2025

This directory contains the midterm exam for the Machine Learning for Computer Systems course (Fall 2025).

## Files

- `exam.tex` - Main LaTeX file
- `questions.tex` - Exam questions with solutions
- `instructions.tex` - Standard exam instructions
- `feamster.sty` - LaTeX style package (required for compilation)
- `Makefile` - Build system for generating PDFs

## Building the Exam

### Prerequisites
- LaTeX distribution (TeX Live, MacTeX, etc.)
- `pdflatex` command available in your PATH

### Commands

```bash
# Build student version (no solutions)
make exam

# Build solution version (with solutions)
make solution

# Build both versions (default)
make all

# Clean auxiliary files (.aux, .log, etc.)
make clean

# Remove all generated files including PDFs
make distclean
```

## Output Files

- `exam.pdf` - Student version (no solutions shown)
- `exam-solution.pdf` - Solution version (solutions shown in answer boxes)

## Coverage

This exam covers material from Meetings 1-8:
- Motivating applications (video QoE, security, resource allocation)
- Data acquisition (passive vs. active measurement, flows, packet captures)
- Machine learning pipeline and data quality
- Model training and evaluation (bias-variance tradeoff, evaluation metrics)
- Cross-validation and model drift

## Point Distribution

- Motivating Applications: 6 points
- Security Applications: 8 points
- Data Acquisition: 11 points
- ML Pipeline and Data Quality: 6 points
- Model Training and Evaluation: 9 points
- Cross-Validation and Model Drift: 7 points
- Feedback: 3 points

**Total: 50 points** (4 pages)

## Question Format

The exam includes a mix of:
- **Multiple choice** ("select all that apply" format) for quick assessment
- **Yes/No with explanation** - allows quick grading (check yes/no first) with partial credit opportunity in explanations
- **Short answer** - tests deeper understanding

This format allows efficient grading: quickly scan yes/no and multiple choice answers, then read explanations only where needed for partial credit.
