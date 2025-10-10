#!/bin/bash

KEY_ID="acmark-ai"
SECRET="VYNZrsOfdVB390E+M41dxy5fQ7RjKiaPKtWyrFUrR0MZOvtttrdb0jd0Kk/wGJrC"
BASE_URL="http://192.168.240.9:2000"

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
NONCE=$(uuidgen)
# BODY='{"action":"create","module":"meetings","parameters":{"name":"Kontrola smlouvy","related_to":"Michal Zagor","related_to_id":"31c67ec6-81e3-6e59-1bb5-6683e1a8b7e2","related_module":"contacts"},"metadata":{"date":"2025-08-06","time":"09:00","duration":60,"participants":["Michal Zagor"],"location":"Online meeting"},"message_to_user":"Schůzka ..."}'
# BODY='{"command": { "action": "create", "module": "meetings", "parameters": { "name": "Schůzka s Janem Piknou", "related_to": "contact", "related_to_id": "1092372d-2989-27b3-1f98-5523d7869c50", "related_module": "contacts" }, "metadata": { "date": "2025-09-05", "time": "10:45", "duration": 60, "participants": [ "Jan Pikna" ], "location": "Invex", "note": "Projednání analýzy produktu" }, "message_to_user": "Schůzka byla vytvořena." }, "user": {"id": "28", "username": "jkovar"}}'
BODY='{"command": {    "action": "create",    "module": "meetings",    "parameters": {        "name": null,        "related_to": "7d669c41-7d80-1156-1dc4-559245d3331d",        "related_to_id": "7d669c41-7d80-1156-1dc4-559245d3331d",        "related_module": "accounts"    },    "metadata": {        "date": "2025-10-01",        "time": "12:30",        "duration": 60,        "participants": [            "61cbb91d-9d8f-88c1-94cb-5f5237151e51",            "662aa625-4f7a-dfaf-516c-5f5236670da3",            "9072241f-ce0d-2b65-15e7-5534c0fa4bd5"        ],        "location": null    },    "message_to_user": "Schůzka s všemi kontakty firmy UNIONISTA KONSTRUKTIVNÍ a.s. je plánována na úterý 2025-10-01 v 12:30."}, "user": {"id": "28", "username": "jkovar"}}'

BASE="${TS}|${NONCE}|${BODY}"
SIG=$(printf '%s' "$BASE" | openssl dgst -sha256 -hmac "$SECRET" -binary | openssl base64 -A)

curl -i "$BASE_URL/ai/v1/command" \
  -H "Content-Type: application/json" \
  -H "Authorization: HMAC keyId=${KEY_ID}, signature=${SIG}" \
  -H "X-Timestamp: ${TS}" \
  -H "X-Nonce: ${NONCE}" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "X-Request-Id: $(uuidgen)" \
  -H "X-Command-Schema: crm.v1" \
  --data "$BODY"
