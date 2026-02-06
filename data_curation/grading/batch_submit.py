"""OpenAI Batch API operations: upload, submit, check status, retrieve results.

Usage:
    python -m data_curation.grading.batch_submit upload --file batch_grading_requests.jsonl
    python -m data_curation.grading.batch_submit submit --file-id file-abc123
    python -m data_curation.grading.batch_submit check --batch-id batch_abc123
    python -m data_curation.grading.batch_submit retrieve --batch-id batch_abc123 --output batch_results.jsonl

Requires: OPENAI_API_KEY environment variable.
"""

import argparse
import os
import sys

from openai import OpenAI


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def cmd_upload(args):
    """Upload a JSONL file for batch processing."""
    client = get_client()
    file = client.files.create(
        file=open(args.file, "rb"),
        purpose="batch"
    )
    print(f"Uploaded: {file.id}")
    print(f"Use this file ID with: submit --file-id {file.id}")


def cmd_submit(args):
    """Submit a batch job."""
    client = get_client()
    batch = client.batches.create(
        input_file_id=args.file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": args.description}
    )
    print(f"Batch ID: {batch.id}")
    print(f"Status:   {batch.status}")
    print(f"Check with: check --batch-id {batch.id}")


def cmd_check(args):
    """Check batch job status."""
    client = get_client()
    batch = client.batches.retrieve(args.batch_id)
    print(f"Status:          {batch.status}")
    print(f"Request counts:  {batch.request_counts}")
    print(f"Output file ID:  {batch.output_file_id}")
    print(f"Error file ID:   {batch.error_file_id}")


def cmd_retrieve(args):
    """Download batch results to a JSONL file."""
    client = get_client()
    batch = client.batches.retrieve(args.batch_id)

    print(f"Status: {batch.status}")
    print(f"Request counts: {batch.request_counts}")

    if batch.output_file_id is None:
        print("Batch not completed yet (no output_file_id). Wait until status=completed.")
        return

    resp = client.files.content(batch.output_file_id)

    with open(args.output, "wb") as f:
        f.write(resp.read())

    print(f"Saved results to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API operations")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_upload = sub.add_parser("upload", help="Upload JSONL file for batch processing")
    p_upload.add_argument("--file", required=True, help="JSONL file to upload")
    p_upload.set_defaults(func=cmd_upload)

    p_submit = sub.add_parser("submit", help="Submit a batch job")
    p_submit.add_argument("--file-id", required=True, help="Uploaded file ID")
    p_submit.add_argument("--description", default="Safety grading batch",
                          help="Batch description metadata")
    p_submit.set_defaults(func=cmd_submit)

    p_check = sub.add_parser("check", help="Check batch job status")
    p_check.add_argument("--batch-id", required=True, help="Batch job ID")
    p_check.set_defaults(func=cmd_check)

    p_retrieve = sub.add_parser("retrieve", help="Download batch results")
    p_retrieve.add_argument("--batch-id", required=True, help="Batch job ID")
    p_retrieve.add_argument("--output", default="batch_results.jsonl", help="Output JSONL path")
    p_retrieve.set_defaults(func=cmd_retrieve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
