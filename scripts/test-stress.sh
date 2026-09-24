#!/usr/bin/env bash
set -euo pipefail

# Configuration
NUM_REQUESTS="${1:-5}"
BASE_URL="http://127.0.0.1:8000"
AUTH_PATH="/v1/systemone"

if [[ -t 1 ]]; then
  GREEN='\033[0;32m'
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  CYAN='\033[0;36m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  GREEN=''
  RED=''
  BLUE=''
  CYAN=''
  BOLD=''
  NC=''
fi

PAYLOAD='{
  "model": "von-latest",
  "state": "Payment gateway reports timeout on charge authorizations. Urgent.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What is the operational nature of this ticket?",
      "criteria": {
        "payment_failure": "Failures processing charges, gateway timeouts",
        "access_issue": "Login, SSO, authentication, or permission errors"
      }
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "Does the request require immediate intervention?"
    },
    "severity": {
      "type": "score",
      "instructions": "Rate incident severity.",
      "criteria": ["Low", "Medium", "High", "Critical"]
    }
  }
}'

get_time() {
  date +%s.%N
}

calculate_duration() {
  python3 -c "print(round($2 - $1, 3))"
}

echo -e "${CYAN}======================================================================${NC}"
echo -e "${BOLD}  VON-SERVER STRESS & PERFORMANCE TESTS (in-pod / no auth)${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Target URL:      ${BOLD}$BASE_URL${NC}"
echo -e "Endpoint:        ${BOLD}$AUTH_PATH${NC}"
echo -e "Requests / Mode: ${BOLD}$NUM_REQUESTS${NC}"
echo

# ----------------------------------------------------------------------
# Mode 1: Sequential
# ----------------------------------------------------------------------
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo -e "${BOLD}[1] SEQUENTIAL MODE${NC}"
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo "Sending $NUM_REQUESTS requests one at a time..."

m1_start=$(get_time)
m1_success=0

for ((i=1; i<=NUM_REQUESTS; i++)); do
  echo -n "  Request #$i: "
  status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL$AUTH_PATH" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

  if [[ "$status" == "200" || "$status" == "422" ]]; then
    echo -e "${GREEN}PASS${NC} ($status)"
    m1_success=$((m1_success + 1))
  else
    echo -e "${RED}FAIL${NC} ($status)"
  fi
done

m1_end=$(get_time)
m1_duration=$(calculate_duration "$m1_start" "$m1_end")
m1_avg=$(python3 -c "print(round($m1_duration / $NUM_REQUESTS, 3))")

echo
echo -e "  Success Rate: ${BOLD}$m1_success / $NUM_REQUESTS${NC}"
echo -e "  Total Time:   ${BOLD}${m1_duration}s${NC}"
echo -e "  Avg Latency:  ${BOLD}${m1_avg}s / request${NC}"
echo

# ----------------------------------------------------------------------
# Mode 2: Concurrent (with response body capture)
# ----------------------------------------------------------------------
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo -e "${BOLD}[2] CONCURRENT MODE${NC}"
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo "Launching $NUM_REQUESTS requests in parallel..."

m2_start=$(get_time)
tmp_dir=$(mktemp -d)
pids=()

for ((i=1; i<=NUM_REQUESTS; i++)); do
  (
    body=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL$AUTH_PATH" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD")
    status=$(echo "$body" | tail -n1)
    response=$(echo "$body" | head -n -1)
    echo "$status" > "$tmp_dir/status_$i"
    echo "$response" > "$tmp_dir/body_$i"
  ) &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

m2_end=$(get_time)
m2_duration=$(calculate_duration "$m2_start" "$m2_end")

m2_success=0
for ((i=1; i<=NUM_REQUESTS; i++)); do
  status=$(cat "$tmp_dir/status_$i" 2>/dev/null || echo "failed")
  body=$(cat "$tmp_dir/body_$i" 2>/dev/null || echo "(no body)")
  if [[ "$status" == "200" || "$status" == "422" ]]; then
    echo -e "  Request #$i: ${GREEN}PASS${NC} ($status)"
    m2_success=$((m2_success + 1))
  else
    echo -e "  Request #$i: ${RED}FAIL${NC} ($status)"
    echo -e "    Body: $body"
  fi
done
rm -rf "$tmp_dir"

echo
echo -e "  Success Rate: ${BOLD}$m2_success / $NUM_REQUESTS${NC}"
echo -e "  Total Time:   ${BOLD}${m2_duration}s${NC}"
echo

# ----------------------------------------------------------------------
# Mode 3: Single curl chained via --next
# ----------------------------------------------------------------------
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo -e "${BOLD}[3] SINGLE CURL MODE (chained via --next)${NC}"
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo "Executing single curl with $NUM_REQUESTS chained requests..."

cmd=("curl")
for ((i=1; i<=NUM_REQUESTS; i++)); do
  if (( i > 1 )); then
    cmd+=("--next")
  fi
  cmd+=(
    "-s"
    "-o" "/dev/null"
    "-w" "%{http_code}\n"
    "-X" "POST"
    "$BASE_URL$AUTH_PATH"
    "-H" "Content-Type: application/json"
    "-d" "$PAYLOAD"
  )
done

m3_start=$(get_time)
m3_success=0
m3_total=0

while read -r status; do
  if [[ -n "$status" ]]; then
    m3_total=$((m3_total + 1))
    if [[ "$status" == "200" || "$status" == "422" ]]; then
      m3_success=$((m3_success + 1))
    fi
  fi
done < <( "${cmd[@]}" )

m3_end=$(get_time)
m3_duration=$(calculate_duration "$m3_start" "$m3_end")
m3_avg=$(python3 -c "print(round($m3_duration / $NUM_REQUESTS, 3))")

echo "  Done."
echo
echo -e "  Success Rate: ${BOLD}$m3_success / $m3_total${NC}"
echo -e "  Total Time:   ${BOLD}${m3_duration}s${NC}"
echo -e "  Avg Latency:  ${BOLD}${m3_avg}s / request${NC}"
echo

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo -e "${CYAN}======================================================================${NC}"
echo -e "${BOLD}  SUMMARY${NC}"
echo -e "${CYAN}======================================================================${NC}"
printf "  %-25s %-15s %-15s\n" "Test Mode" "Total Time" "Success Rate"
echo -e "  --------------------------------------------------------------------"
printf "  %-25s %-15s %-15s\n" "Sequential" "${m1_duration}s" "$m1_success/$NUM_REQUESTS"
printf "  %-25s %-15s %-15s\n" "Concurrent" "${m2_duration}s" "$m2_success/$NUM_REQUESTS"
printf "  %-25s %-15s %-15s\n" "Single Curl" "${m3_duration}s" "$m3_success/$NUM_REQUESTS"
echo -e "${CYAN}======================================================================${NC}"
