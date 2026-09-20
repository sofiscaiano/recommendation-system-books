from __future__ import annotations

import sqlite3

from quelibroleo.data import load_dataset, observed_catalog


def test_load_dataset_quarantines_bad_dates_and_orphans(tmp_path) -> None:
    path = tmp_path / "tiny.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE interacciones (
            id_lector TEXT,
            id_libro TEXT,
            fecha TEXT,
            rating INT,
            PRIMARY KEY(id_lector, id_libro)
        );
        CREATE TABLE libros (
            id_libro TEXT PRIMARY KEY,
            titulo TEXT,
            autor TEXT,
            genero TEXT,
            editorial TEXT,
            anio_edicion,
            isbn,
            resumen TEXT,
            img_src TEXT
        );
        CREATE TABLE lectores (
            id_lector TEXT PRIMARY KEY,
            nombre TEXT,
            genero TEXT,
            vive_en,
            nacimiento
        );
        INSERT INTO libros VALUES
            ('b1', 'Libro', 'Autora', 'Género', 'Editora', 2020, NULL, 'Texto', NULL),
            ('unused', '', '', '', '', NULL, NULL, '', NULL);
        INSERT INTO lectores VALUES ('u1', 'Nombre', '-', '', '');
        INSERT INTO interacciones VALUES
            ('u1', 'b1', '01-02-2020', 8),
            ('missing-user', 'missing-book', '02-02-2020', 7),
            ('u1', 'bad-date-book', 'fecha mala', 5);
        """
    )
    connection.commit()
    connection.close()

    bundle = load_dataset(path)

    assert bundle.report.invalid_date_count == 1
    assert bundle.report.orphan_user_count == 1
    assert bundle.report.orphan_item_count == 1
    assert len(bundle.modeled_interactions) == 1
    catalog = observed_catalog(bundle.modeled_interactions, bundle.books)
    assert catalog["id_libro"].tolist() == ["b1"]
