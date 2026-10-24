# Makefile for ML Systems Midterm Exam
# Generates both student version (no solutions) and solution version

.PHONY: all clean exam solution

# Default target: build both versions
all: exam solution

# Student version (no solutions)
exam: exam.pdf

exam.pdf: exam.tex instructions.tex questions.tex feamster.sty
	@echo "Building student version (no solutions)..."
	pdflatex -synctex=1 exam.tex
	pdflatex -synctex=1 exam.tex
	@echo "Student version created: exam.pdf"

# Solution version (with solutions)
solution: exam-solution.pdf

exam-solution.pdf: exam.tex instructions.tex questions.tex feamster.sty
	@echo "Building solution version..."
	# Create temporary file with solution option enabled
	sed 's/\\usepackage\[\]{feamster}/\\usepackage[solution]{feamster}/' exam.tex > exam-solution.tex
	pdflatex -synctex=1 exam-solution.tex
	pdflatex -synctex=1 exam-solution.tex
	@echo "Solution version created: exam-solution.pdf"

# Clean up auxiliary files and PDFs
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f exam-solution.tex
	rm -f exam.pdf exam-solution.pdf

# Clean everything including PDFs (alias for clean)
distclean: clean
