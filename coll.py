import glob, os
import string
from stemming.porter2 import stem

class BowDoc:
    """Bag-of-words representation of a document.

    The document has an ID, and an iterable list of terms with their
    frequencies."""

    def __init__(self, docid):
        """Constructor.

        Set the ID of the document, and initiate an empty term dictionary.
        Call add_term to add terms to the dictionary."""
        self.docid = docid
        self.terms = {}
        self.doc_len = 0

    def add_term(self, term):
        """Add a term occurrence to the BOW representation.

        This should be called each time the term occurs in the document."""
        try:
            self.terms[term] += 1
        except KeyError:  
            self.terms[term] = 1

    def get_term_count(self, term):
        """Get the term occurrence count for a term.

        Returns 0 if the term does not appear in the document."""
        try:
            return self.terms[term]
        except KeyError:
            return 0

    def get_term_freq_dict(self):
        """Return dictionary of term:freq pairs."""
        return self.terms

    def get_term_list(self):
        """Get sorted list of all terms occurring in the document."""
        return sorted(self.terms.keys())

    def get_docid(self):
        """Get the ID of the document."""
        return self.docid

    def __iter__(self):
        """Return an ordered iterator over term--frequency pairs.

        Each element is a (term, frequency) tuple.  They are iterated
        in term's frequency descending order."""
        return iter(sorted(self.terms.items(), key=lambda x: x[1],reverse=True))
        """Or in term alphabetical order:
        return iter(sorted(self.terms.iteritems()))"""
    def get_doc_len(self):
        return self.doc_len

    def set_doc_len(self, doc_len):
        self.doc_len = doc_len

class BowColl:
    """Collection of BOW documents."""

    def __init__(self):
        """Constructor.

        Creates an empty collection."""
        self.docs = {}

    def add_doc(self, doc):
        """Add a document to the collection."""
        self.docs[doc.get_docid()] = doc

    def get_doc(self, docid):
        """Return a document by docid.

        Will raise a KeyError if there is no document with that ID."""
        return self.docs[docid]

    def get_docs(self):
        """Get the full list of documents.

        Returns a dictionary, with docids as keys, and docs as values."""
        return self.docs

    def inorder_iter(self):
        """Return an ordered iterator over the documents.
        
        The iterator will traverse the collection in docid order.  Modifying
        the collection while iterating over it leads to undefined results.
        Each element is a document; to find the id, call doc.get_docid()."""
        return BowCollInorderIterator(self)

    def get_num_docs(self):
        """Get the number of documents in the collection."""
        return len(self.docs)

    def __iter__(self):
        """Iterator interface.

        See inorder_iter."""
        return self.inorder_iter()

class BowCollInorderIterator:
    """Iterator over a collection."""

    def __init__(self, coll):
        """Constructor.
        
        Takes the collection we're going to iterator over as sole argument."""
        self.coll = coll
        self.keys = sorted(coll.get_docs().keys())
        self.i = 0

    def __iter__(self):
        """Iterator interface."""
        return self

    def next(self):
        """Get next element."""
        if self.i >= len(self.keys):
            raise StopIteration
        doc = self.coll.get_doc(self.keys[self.i])
        self.i += 1
        return doc

def parse_rcv_coll(inputpath, stop_words):
    """Parse an RCV1 data files into a collection.

    inputpath is the folder name of the RCV1 data files.  The parsed collection
    is returned.  NOTE the function performs very limited error checking."""
    #stopwords = open('common-english-words.txt', 'r')
    
    coll = BowColl()    
    os.chdir(inputpath)
    for file_ in glob.glob("*.xml"):
        curr_doc = None
        start_end = False
        word_count = 0
        for line in open(file_):
            line = line.strip()
            if(start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            curr_doc = BowDoc(docid)
                            break
                    continue    
                if line.startswith("<text>"):
                    start_end = True  
            elif line.startswith("</text>"):
                break
            elif curr_doc is not None:
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

                for term in line.split():
                    word_count += 1
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        curr_doc.add_term(term)
        if curr_doc is not None:
            curr_doc.set_doc_len(word_count)
            coll.add_doc(curr_doc)
    return coll


