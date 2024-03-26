.PHONY: clean

target=expose

$(target).pdf: $(target).tex
	pdflatex $(target)
	biber $(target)
	pdflatex $(target)
	pdflatex $(target)

clean:
	rm -rf *.log *.aux *.pdf *.xml *.blg *.bcf *.bbl
