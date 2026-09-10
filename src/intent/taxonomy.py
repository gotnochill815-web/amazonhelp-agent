
"""
AmazonHelp Intent Taxonomy

The taxonomy was derived from exploratory semantic clustering of
5,000 AmazonHelp customer messages, followed by manual consolidation
according to operationally meaningful support workflows.

This taxonomy is used for:
- intent classification
- golden-set annotation
- evaluation
- error analysis

Important:
Escalation is NOT part of the intent taxonomy. It is a separate
decision layer because the same intent can sometimes be safely
handled automatically and sometimes require human intervention.
"""


INTENTS = {
    "delivery_status": {
        "description": (
            "Questions about order tracking, dispatch status, "
            "estimated arrival, or where an order currently is."
        ),
        "include_when": [
            "Customer asks where the order is.",
            "Customer asks for tracking information.",
            "Customer asks whether an order has shipped.",
            "Customer asks for the expected arrival date."
        ],
        "exclude_when": [
            "The customer explicitly reports a late delivery.",
            "The customer says the package was marked delivered but was not received.",
            "The issue is specifically about a preorder release date."
        ],
    },

    "delivery_problem": {
        "description": (
            "Problems with physical delivery fulfillment, including "
            "late, failed, missing, wrong-location, or damaged delivery."
        ),
        "include_when": [
            "Package is late or overdue.",
            "Package is marked delivered but was not received.",
            "Package was delivered to the wrong location.",
            "Delivery attempt failed.",
            "Package or items were damaged during delivery."
        ],
        "exclude_when": [
            "Customer is only asking for normal tracking/status.",
            "Issue is specifically about cancelling or modifying an order."
        ],
    },

    "order_issue": {
        "description": (
            "Problems involving the order itself, such as cancellation, "
            "modification, unexpected cancellation, or incorrect order details."
        ),
        "include_when": [
            "Customer wants to cancel an order.",
            "Order was unexpectedly cancelled.",
            "Customer wants to change an order.",
            "Order details are incorrect."
        ],
        "exclude_when": [
            "The main issue is physical delivery.",
            "The main issue is a refund after a completed return."
        ],
    },

    "preorder_issue": {
        "description": (
            "Problems involving pre-orders, release dates, dispatch timing, "
            "or expected delivery around a product release."
        ),
        "include_when": [
            "A pre-order has not shipped.",
            "A pre-order is late relative to release day.",
            "Customer asks about release-day delivery.",
            "Pre-order availability or dispatch is unclear."
        ],
        "exclude_when": [
            "The order is a normal non-preorder order.",
            "The primary issue is a completed-order refund."
        ],
    },

    "returns_replacements": {
        "description": (
            "Returns, exchanges, replacements, and return-process problems."
        ),
        "include_when": [
            "Customer wants to return an item.",
            "Customer wants a replacement or exchange.",
            "Return label or return instructions are not working.",
            "Replacement was not received."
        ],
        "exclude_when": [
            "The main issue is waiting for a refund after a return.",
            "The main issue is a payment or billing charge."
        ],
    },

    "refund_issue": {
        "description": (
            "Problems involving refunds, including missing, delayed, "
            "incorrect, or unexpectedly initiated refunds."
        ),
        "include_when": [
            "Customer has not received an expected refund.",
            "Refund is taking too long.",
            "Customer received an incorrect refund.",
            "Customer wants to change a refund/replacement outcome."
        ],
        "exclude_when": [
            "Customer is asking how to initiate a return.",
            "The main issue is an active payment charge rather than a refund."
        ],
    },

    "payment_billing": {
        "description": (
            "Payment failures, payment methods, billing problems, "
            "unexpected charges, duplicate charges, or payment processing."
        ),
        "include_when": [
            "Payment was declined or failed.",
            "Customer cannot use a payment method.",
            "Customer reports an unexpected charge.",
            "Customer reports duplicate or incorrect billing."
        ],
        "exclude_when": [
            "The main issue is account compromise/security.",
            "The charge is specifically a normal Prime membership question."
        ],
    },

    "account_security": {
        "description": (
            "Account access, login, password, unauthorized account changes, "
            "phishing, fraud, or account-security concerns."
        ),
        "include_when": [
            "Account is locked.",
            "Customer cannot log in.",
            "Password or email changed without authorization.",
            "Customer reports phishing or unauthorized access.",
            "Customer believes the account was compromised."
        ],
        "exclude_when": [
            "The issue is only a normal payment problem.",
            "The issue is only a Prime membership charge."
        ],
    },

    "prime_membership": {
        "description": (
            "Amazon Prime membership, subscription, benefits, "
            "membership activation, or Prime-specific service questions."
        ),
        "include_when": [
            "Customer asks about Prime membership.",
            "Prime benefits are missing or not working.",
            "Customer has a Prime subscription question.",
            "Customer reports a Prime membership charge."
        ],
        "exclude_when": [
            "The complaint is only about physical delivery timing and "
            "contains no meaningful membership issue."
        ],
    },

    "digital_product_support": {
        "description": (
            "Technical support for Amazon digital products and services, "
            "including Kindle, Echo/Alexa, Fire TV, Prime Video, apps, "
            "content playback, and related functionality."
        ),
        "include_when": [
            "Kindle is not working.",
            "Echo/Alexa functionality is broken.",
            "Fire TV problems are reported.",
            "Prime Video or digital content will not play.",
            "Amazon app has a technical error."
        ],
        "exclude_when": [
            "The issue is physical product delivery.",
            "The issue is account access without a digital-product component."
        ],
    },

    "general_support": {
        "description": (
            "General support, follow-up, complaints, dissatisfaction, "
            "or messages that do not map cleanly to another operational intent."
        ),
        "include_when": [
            "Customer asks for general help.",
            "Customer is following up on an unresolved support case.",
            "Customer complains about support quality.",
            "Message is ambiguous and lacks enough information for another intent.",
            "Message is conversational/noisy but is still directed at support."
        ],
        "exclude_when": [
            "A specific operational intent can be confidently identified."
        ],
    },
}


INTENT_NAMES = list(INTENTS.keys())


def get_intent_names() -> list[str]:
    """Return the canonical intent names."""
    return INTENT_NAMES.copy()


def get_intent_definition(intent: str) -> dict:
    """Return a single intent definition."""
    if intent not in INTENTS:
        raise KeyError(f"Unknown intent: {intent}")

    return INTENTS[intent]


def validate_intent(intent: str) -> bool:
    """Return True when an intent belongs to the taxonomy."""
    return intent in INTENTS


# Annotation policy for single-label evaluation.
ANNOTATION_POLICY = {
    "primary_intent_rule": (
        "Assign the intent corresponding to the customer's primary "
        "actionable support request."
    ),
    "multi_intent_rule": (
        "When multiple issues are present, label the issue that best "
        "explains what the customer wants the support agent to resolve."
    ),
    "escalation_separation": (
        "Escalation is labelled independently from intent."
    ),
    "ambiguous_rule": (
        "Use general_support when the message does not contain enough "
        "information to reliably assign a more specific operational intent."
    ),
}


if __name__ == "__main__":
    print(f"Number of intents: {len(INTENTS)}")
    for name in INTENTS:
        print(f"- {name}")
