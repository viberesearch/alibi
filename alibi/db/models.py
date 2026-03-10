"""Pydantic models for Alibi entities."""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Enums for constrained fields


class SpaceType(str, Enum):
    """Type of space (private or shared)."""

    PRIVATE = "private"
    SHARED = "shared"


class DocumentType(str, Enum):
    """Type of document artifact."""

    RECEIPT = "receipt"
    INVOICE = "invoice"
    STATEMENT = "statement"
    WARRANTY = "warranty"
    POLICY = "policy"
    CONTRACT = "contract"
    PAYMENT_CONFIRMATION = "payment_confirmation"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Processing status of an artifact."""

    PENDING = "pending"
    PROCESSED = "processed"
    VERIFIED = "verified"
    ERROR = "error"


class ItemStatus(str, Enum):
    """Status of an owned item/asset."""

    ACTIVE = "active"
    SOLD = "sold"
    DISPOSED = "disposed"
    RETURNED = "returned"
    LOST = "lost"


# V2 Enums (adapted from ma-engine patterns)


class RecordType(str, Enum):
    """Unified type for all alibi records."""

    PAYMENT = "payment"
    PURCHASE = "purchase"
    REFUND = "refund"
    INVOICE = "invoice"
    STATEMENT = "statement"
    WARRANTY = "warranty"
    INSURANCE = "insurance"
    GUARANTEE = "guarantee"
    CLAIM = "claim"
    CLAIM_RESOLUTION = "claim_resolution"
    SUBSCRIPTION = "subscription"
    CONTRACT = "contract"
    TAX_DOCUMENT = "tax_document"
    MERCHANT = "merchant"
    ACCOUNT = "account"


class DataType(str, Enum):
    """Temporal classification (from ma-engine)."""

    ACTUAL = "actual"
    PROJECTED = "projected"
    TARGET = "target"


class UnitType(str, Enum):
    """Measurement units for line items."""

    GRAM = "g"
    KILOGRAM = "kg"
    POUND = "lb"
    OUNCE = "oz"
    MILLILITER = "ml"
    LITER = "l"
    GALLON = "gal"
    PIECE = "pcs"
    PACK = "pack"
    KWH = "kWh"
    METER = "m"
    SQ_METER = "sqm"
    CUBIC_METER = "cbm"
    HOUR = "hr"
    MINUTE = "min"
    OTHER = "other"


class TaxType(str, Enum):
    """Tax classification."""

    VAT = "vat"
    SALES_TAX = "sales_tax"
    GST = "gst"
    EXEMPT = "exempt"
    INCLUDED = "included"
    NONE = "none"


class Tier(str, Enum):
    """Disclosure tier (from ma-engine). T0=masked, T4=exact."""

    T0 = "0"
    T1 = "1"
    T2 = "2"
    T3 = "3"
    T4 = "4"


class DisplayType(str, Enum):
    """How to present a value at a given tier."""

    EXACT = "exact"
    RANGE = "range"
    ROUNDED = "rounded"
    MASKED = "masked"
    HIDDEN = "hidden"


class FieldType(str, Enum):
    """Semantic type of a field value."""

    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    TEXT = "text"
    DATE = "date"
    BOOLEAN = "boolean"
    WEIGHT = "weight"
    VOLUME = "volume"
    COUNT = "count"
    ENERGY = "energy"
    DISTANCE = "distance"
    DURATION = "duration"
    AREA = "area"
    NUMBER = "number"


# ---------------------------------------------------------------------------
# Atom-Cloud-Fact enums (v2 schema)
# ---------------------------------------------------------------------------


class AtomType(str, Enum):
    """Type of extracted observation atom."""

    ITEM = "item"
    PAYMENT = "payment"
    VENDOR = "vendor"
    DATETIME = "datetime"
    AMOUNT = "amount"
    TAX = "tax"


class BundleType(str, Enum):
    """Structural grouping of atoms from one document."""

    BASKET = "basket"
    PAYMENT_RECORD = "payment_record"
    INVOICE = "invoice"
    STATEMENT_LINE = "statement_line"


class BundleAtomRole(str, Enum):
    """Role of an atom within a bundle."""

    BASKET_ITEM = "basket_item"
    TOTAL = "total"
    SUBTOTAL = "subtotal"
    VENDOR_INFO = "vendor_info"
    PAYMENT_INFO = "payment_info"
    EVENT_TIME = "event_time"
    TAX_DETAIL = "tax_detail"


class CloudStatus(str, Enum):
    """Status of a probabilistic cloud cluster."""

    FORMING = "forming"
    COLLAPSED = "collapsed"
    DISPUTED = "disputed"


class CloudMatchType(str, Enum):
    """How a bundle was matched into a cloud."""

    EXACT_AMOUNT = "exact_amount"
    NEAR_AMOUNT = "near_amount"
    SUM_OF_PARTS = "sum_of_parts"
    VENDOR_DATE = "vendor+date"
    ITEM_OVERLAP = "item_overlap"
    MANUAL = "manual"


class FactType(str, Enum):
    """Type of confirmed fact."""

    PURCHASE = "purchase"
    REFUND = "refund"
    SUBSCRIPTION_PAYMENT = "subscription_payment"


class FactStatus(str, Enum):
    """Confirmation status of a fact."""

    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    NEEDS_REVIEW = "needs_review"


# Base model with common fields


class TimestampedModel(BaseModel):
    """Base model with created_at timestamp."""

    created_at: datetime = Field(default_factory=datetime.now)


# Core entity models


class User(TimestampedModel):
    """A user of the system."""

    id: str
    name: str


class Space(TimestampedModel):
    """A space for organizing data (private or shared)."""

    id: str
    name: str
    type: SpaceType
    owner_id: str


class Artifact(TimestampedModel):
    """In-memory document representation for matching and note generation.

    Not backed by a database table — used as a data transfer object
    in matching/duplicates.py, obsidian/notes.py, and mycelium/notes.py.
    """

    id: str
    space_id: str = "default"
    type: DocumentType = DocumentType.OTHER
    file_path: str = ""
    file_hash: str = ""
    perceptual_hash: Optional[str] = None
    vendor: Optional[str] = None
    vendor_id: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_phone: Optional[str] = None
    vendor_website: Optional[str] = None
    vendor_vat: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    record_type: Optional[str] = None
    document_id: Optional[str] = None
    document_date: Optional[date] = None
    amount: Optional[Decimal] = None
    currency: str = "EUR"
    raw_text: Optional[str] = None
    extracted_data: Optional[dict[str, Any]] = None
    status: DocumentStatus = DocumentStatus.PENDING
    transaction_time: Optional[str] = None


class Item(TimestampedModel):
    """An owned item/asset."""

    id: str
    space_id: str
    name: str
    category: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    purchase_date: Optional[date] = None
    purchase_price: Optional[Decimal] = None
    current_value: Optional[Decimal] = None
    currency: str = "EUR"
    status: ItemStatus = ItemStatus.ACTIVE
    warranty_expires: Optional[date] = None
    warranty_type: Optional[str] = None
    insurance_covered: bool = False
    note_path: Optional[str] = None
    modified_at: Optional[datetime] = None
    created_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Atom-Cloud-Fact models (v2 schema)
# ---------------------------------------------------------------------------


class Document(TimestampedModel):
    """Source document (v2 — replaces Artifact for storage)."""

    id: str
    file_path: str
    file_hash: str  # SHA-256
    perceptual_hash: Optional[str] = None  # dHash for images
    raw_extraction: Optional[dict[str, Any]] = None  # Full LLM output
    source: Optional[str] = None  # Entry point: telegram, api, cli, watcher, mcp
    user_id: Optional[str] = None  # User who submitted the document
    yaml_hash: Optional[str] = None  # SHA-256 of .alibi.yaml for correction detection
    yaml_path: Optional[str] = (
        None  # Path to .alibi.yaml in yaml_store (None = sidecar)
    )
    ingested_at: datetime = Field(default_factory=datetime.now)


class Atom(TimestampedModel):
    """Individual extracted observation from a document."""

    id: str
    document_id: str
    atom_type: AtomType
    data: dict[str, Any]  # Type-specific payload (normalized)
    confidence: Decimal = Decimal("1.0")


class Bundle(TimestampedModel):
    """Structural atom group from one document."""

    id: str
    document_id: str
    bundle_type: BundleType
    cloud_id: str | None = None  # Authoritative cloud assignment


class BundleAtom(BaseModel):
    """Link between a bundle and an atom with role."""

    bundle_id: str
    atom_id: str
    role: BundleAtomRole


class Cloud(TimestampedModel):
    """Probabilistic cluster of bundles across documents."""

    id: str
    status: CloudStatus = CloudStatus.FORMING
    confidence: Decimal = Decimal("0.0")


class CloudBundle(BaseModel):
    """Link between a cloud and a bundle with match metadata."""

    cloud_id: str
    bundle_id: str
    match_type: CloudMatchType
    match_confidence: Decimal = Decimal("0.0")


class Fact(TimestampedModel):
    """Collapsed cloud — confirmed real-world event."""

    id: str
    cloud_id: str
    fact_type: FactType
    vendor: Optional[str] = None
    vendor_key: Optional[str] = None
    total_amount: Optional[Decimal] = None
    currency: str = "EUR"
    event_date: Optional[date] = None
    payments: Optional[list[dict[str, Any]]] = None
    status: FactStatus = FactStatus.CONFIRMED


class FactItem(TimestampedModel):
    """Denormalized item from an item atom, linked to a fact."""

    id: str
    fact_id: str
    atom_id: str
    name: str
    name_normalized: Optional[str] = None
    comparable_name: Optional[str] = None
    quantity: Decimal = Decimal("1")
    unit: UnitType = UnitType.PIECE
    unit_price: Optional[Decimal] = None
    total_price: Optional[Decimal] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    comparable_unit_price: Optional[Decimal] = None
    comparable_unit: Optional[UnitType] = None
    barcode: Optional[str] = None
    unit_quantity: Optional[Decimal] = None
    tax_rate: Optional[Decimal] = None
    tax_type: TaxType = TaxType.NONE
    enrichment_source: Optional[str] = None
    enrichment_confidence: Optional[float] = None
    product_variant: Optional[str] = None
