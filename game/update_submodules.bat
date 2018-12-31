for /d %%s in (.\*) do (cd %%s
		       call git checkout master && git pull
		       cd ..
		       call git add %%s
		       echo Updated %%s)
call git commit -m "updating game submodules to latest"
timeout /t -1