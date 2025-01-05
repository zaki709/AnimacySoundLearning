include .env

.PHONY: exec
exec:
	docker exec -it ${CONTAINER_NAME} $(CMD)

.PHONY: shell
shell:
	make exec CMD="/bin/bash"

.PHONY: run
run:
	make exec CMD="python src/main.py"


.PHONY: check-env
check-env:
	@echo "---.ENV---" && \
	echo "CONTAINER_NAME: ${CONTAINER_NAME}" &&\
	echo "---libraries---" && \
	docker exec -it ${CONTAINER_NAME} pip list

