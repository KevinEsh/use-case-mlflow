from sqlmodel import Field, SQLModel, Session, create_engine, select

def id_field(table_name: str):
    sequence = sqlalchemy.Sequence(f"{table_name}_id_seq")
    return Field(
        default=None,
        primary_key=True,
        sa_column_args=[sequence],
        sa_column_kwargs={"server_default": sequence.next_value()},
    )

class Contact(SQLModel, table=True):
    id: Optional[int] = id_field("contact")
    email_address: str
    name: Optional[str] = None
    
    sent_emails: List["Email"] = Relationship(back_populates="sender")


class Attachment(SQLModel, table=True):
    id: Optional[int] = id_field("attachment")
    filename: str
    email_id: int = Field(default=None, foreign_key="email.id")

    email: "Email" = Relationship(back_populates="attachments")
 

class RecipientKind(Enum):
    to = "to"
    cc = "cc"
    bcc = "bcc"


class EmailReception(SQLModel, table=True):
    email_id: int = Field(foreign_key="email.id", primary_key=True)
    contact_id: int = Field(foreign_key="contact.id", primary_key=True)
    kind: RecipientKind

    email: "Email" = Relationship(back_populates="recipients")
    contact: "Contact" = Relationship(back_populates="receptions")
 

class Email(SQLModel, table=True):
    id: Optional[int] = id_field("email")
    sender_id: int = Field(foreign_key="contact.id")
    subject: str
    body: str
    
    sender: Contact = Relationship(back_populates="sent_emails")
    attachments: List[Attachment] = Relationship(back_populates="email")
    recipients: List[EmailReception] = Relationship(back_populates="email")

engine = create_engine("duckdb:///test.db")
SQLModel.metadata.create_all(engine)

me = Contact(name="Johannes Köster", email_address="johannes.koester@uni-due.de")
john_doe = Contact(name="John Doe", email_address="john.doe@uni-due.de")

some_email = Email(
    sender=john_doe,
    subject="Hello",
    body="World",
    recipients=[EmailReception(contact=me, kind=RecipientKind.to)],
)

with Session(engine) as session:
    session.add(me)
    session.add(john_doe)
    session.add(some_email)
    session.commit()

    stmt = select(Email).where(Email.sender == john_doe)
    print(session.exec(stmt).first())

    session.exec(
        select(Contact).where(Contact.name == "Johannes Köster")
    ).first().name = "Johannes Simon Köster"
    
    session.commit()
    
    print(
        session.exec(
            select(Contact).where(Contact.name == "Johannes Simon Köster")
        ).first()
    )