---
description: Generate a midterm exam based on course agenda and past exams
---

You are helping create a midterm exam for a Machine Learning for Computer Systems course.

## Task
Create a comprehensive midterm exam that:
1. Covers material from Meetings 1-8 (or as specified in the agenda)
2. Fits on exactly 4 pages (single-sided, not double-sided)
3. Totals exactly 50 points (round number for easy grading)
4. Includes a mix of multiple choice, yes/no with explanation, and short answer questions
5. References concepts from the course assignment(s) and hands-on activities
6. Tests understanding of key concepts without being overly tricky

## Steps
1. Read the course agenda file to understand topics covered in Meetings 1-8
2. Examine 2-3 past midterm exams to understand:
   - Question format and structure
   - Point distribution
   - Mix of question types
   - LaTeX formatting conventions
3. Identify key concepts from each meeting that should be tested
4. Create questions that:
   - Test conceptual understanding, not memorization
   - Include at least one question related to the assignment
   - Cover hands-on activities where applicable
   - Balance difficulty appropriately
5. Generate files in `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/YYYY/`:
   - `questions.tex` - The exam questions with solutions
   - `instructions.tex` - Standard instructions and acknowledgment (without `\prob{1}`)
   - `exam.tex` - Main LaTeX file with `\usepackage[]{feamster}` (no solution flag by default)
   - `Makefile` - Build system that creates both exam.pdf and exam-solution.pdf
   - `README.md` - Documentation for building and using the exam
6. Copy the `feamster.sty` file from a recent exam year to the new directory
7. Build both versions and verify 4 pages, 50 points, no formatting issues
8. **Never commit to git** - files stay in Dropbox/Tmp until instructor manually publishes them

## Key Topics to Cover (adjust based on agenda)
- Motivating applications (video QoE, security, resource allocation)
- Data acquisition (passive vs. active measurement)
- Network data representations (flows, packet captures)
- Feature engineering and preprocessing
- Data quality issues (missing data, non-representative data, irrelevant features, outliers)
- ML pipeline (train/test split, cross-validation, data leakage)
- Model evaluation (accuracy, precision, recall, F1, ROC, confusion matrix)
- Bias-variance tradeoff
- Model drift and distribution shift

## Question Guidelines
- Multiple choice: 3-4 points each, typically "select all that apply"
- Yes/No with explanation: 3-4 points each (allows quick grading of yes/no, then check explanation for partial credit)
- Short answer: 2-5 points each, with answer boxes sized appropriately (2.0-3.0 inches for generous space)
- Include specific examples from class discussions (e.g., TTL overfitting, husky/wolf, Netflix segments)
- End with 3-point feedback section (interest, difficulty, one like, one suggestion)

## LaTeX Format Notes

### Yes/No Questions Format
- For yes/no questions, use `\yesnoyes` (for Yes answer) or `\yesnono` (for No answer)
- Do not use braces: `\yesnoyes` NOT `\yesno{Yes}` or `\yesno{yes}`
- Example:
  ```latex
  \framebox{
  \yesnoyes
  }
  ```
- Follow-up questions should say "Why or why not?" or "Explain why or why not." (NOT "Explain why" which gives away the answer)

### Answer Boxes
- Use `\answerbox{height}{solution text}` for short answer questions
- Solution text must not contain blank lines/paragraph breaks (LaTeX TikZ limitation)
- Separate parts with "(a)" and "(b)" on the same paragraph, not separate paragraphs
- Always put answer boxes on a new line after the question text (add blank line before `\answerbox`)
- Answer box heights: Use 1.0-3.0 inches depending on expected answer length
  - Short explanations: 1.0-1.5 inches
  - Medium explanations: 1.5-2.0 inches
  - Longer explanations: 2.0-3.0 inches
- Use generous box sizes - students need space to write!
- Use `\shortanswerbox{width}{solution text}` for very short answers (like feedback numbers)

### Question Structure
- Use `\prob{N}` to start a question worth N points
- Use `\eprob` to end a question
- Use `\correctanswercircle{}` for correct multiple choice options
- Use `\answercircle{}` for incorrect multiple choice options
- Organize questions into sections with `\section*{Section Name}`
- Add `\vspace*{-0.1in}` after section headers to save space

### Multiple Choice Text Formatting
- Keep option text concise to avoid overflow in two-column layout
- If text is too long for columns, abbreviate or remove parenthetical explanations
- Example: "Precision (what fraction is spam)" can become just "Precision" if it overflows
- Test by compiling - if text spills into adjacent column, shorten it

### Special Characters in LaTeX
- Avoid `&` in answer text (use "and" instead) - causes "Misplaced alignment tab" errors
- This is especially problematic in `\answerbox{}` which uses TikZ nodes

## Critical Point Total Issue
**Note**: The `\prob{N}` macro adds N points to the TotalPoints counter. Past exams incorrectly used `\prob{1}` in instructions.tex for the acknowledgment box, which added an unwanted point to the total.

**Problem**: If you use `\prob{1}` for acknowledgment + 50 points in questions = 51 total points displayed

**Solution**: Do NOT use `\prob{}` or `\eprob` in instructions.tex for the acknowledgment. Instead use plain formatting:
```latex
{\bf 1.} Write your full name in the box to acknowledge the instructions.

\shortanswerbox{3.25}{Nick Feamster}
```

This ensures the exam displays exactly 50 points. The acknowledgment is item #1, but the first actual question in questions.tex will be numbered #2 (which is correct - questions are numbered sequentially across both files).

## Validation and Iteration
After generating the exam:

1. **Build and check page count**:
   ```bash
   make all
   pdftotext exam.pdf - | grep "ML for Systems Midterm" | wc -l
   ```
   Should output: 4

2. **Verify point total**:
   ```bash
   grep '\\prob{' questions.tex | sed 's/.*\\prob{\([0-9]*\)}.*/\1/' | awk '{sum+=$1} END {print sum}'
   ```
   Should output: 50

3. **Check for formatting issues**:
   - Open the PDF and visually inspect for overlapping text
   - Ensure all answer boxes start on a new line
   - Check that multiple choice text doesn't overflow columns
   - Verify yes/no boxes display correctly

4. **Adjust if needed**:
   - If > 4 pages: Remove page breaks between sections, reduce answer box heights, or combine questions
   - If < 4 pages: Increase answer box heights to give students more writing space
   - If text overlaps: Shorten text, especially in multiple choice options
   - If wrong point total: Adjust individual question point values

## Page Break Strategy
- Start with page breaks after major sections: Data Acquisition, ML Pipeline, Model Training
- If exam is too long, remove page breaks to compress
- Keep Cross-Validation section at top of page 4 if possible
- Strategic use of `\vspace*{-0.1in}` after section headers saves space

## Output Location

**IMPORTANT - DO NOT COMMIT TO GIT**: Write files to the Dropbox temporary directory, which is cloud-backed but NOT part of the git repository. Never commit or push exam files to git - the instructor will do this manually after the exam is administered.

Write files to:
- `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/YYYY/questions.tex`
- `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/YYYY/instructions.tex`
- `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/YYYY/exam.tex`
- `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/YYYY/Makefile`
- `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/YYYY/README.md`
- Copy `/Users/feamster/Documents/teaching/ml/ml-systems/docs/midterm/YYYY-1/feamster.sty` to the YYYY directory

This directory structure:
- Is cloud-backed via Dropbox for safety
- Is NOT in the public git repository (repo is at `/Users/feamster/Documents/teaching/ml/ml-systems/`)
- Keeps exam content private until manually published by the instructor

## Building the Exam

The exam uses a single LaTeX file (`exam.tex`) with a solution flag that can be enabled or disabled. The Makefile builds both versions:

**Student version** (no solutions):
```latex
\usepackage[]{feamster}
```

**Solution version** (with solutions):
```latex
\usepackage[solution]{feamster}
```

The Makefile handles this automatically:
- `make exam` - Generates student version (no solutions) as `exam.pdf`
- `make solution` - Generates solution version as `exam-solution.pdf` by using sed to add the solution flag
- `make all` - Generates both versions (default)
- `make clean` - Removes auxiliary files (.aux, .log, etc.)
- `make distclean` - Removes all generated files including PDFs

No need to maintain separate master LaTeX files - the same `exam.tex` is used for both, just compiled differently.

## Before You Start
Ask the user:
1. What year is this exam for?
2. Should you use the standard agenda file location or a different one?
3. Are there any specific topics or questions they want to ensure are included?
4. Which past exams should you reference (suggest looking at 2-3 most recent)?
