include .env

.PHONY: shell
shell:
	docker exec -it ${CONTAINER_NAME} /bin/bash

.PHONY: check-env
check-env:
	@echo "---.ENV---" && \
	echo "CONTAINER_NAME: ${CONTAINER_NAME}" &&\
	echo "---libraries---" && \
	docker exec -it ${CONTAINER_NAME} pip list

