prj_path = out_region3.prj
report:
	find $(prj_path) -type d -name "syn_report" -exec rm -rf {} \;
	mkdir $(prj_path)/syn_report
	cp $(prj_path)/solution1/syn/report/*.rpt $(prj_path)/syn_report
	rm -rf $(prj_path)/syn_report/PE*
	rm -rf $(prj_path)/solution1