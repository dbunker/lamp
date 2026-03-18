d4(o7).
d9(o4).
d1(o4).
d9(o9).
d5(o9).
d8(o9).

d0(X) :- d9(X), not d7(X), d8(X).
d7(X) :- not d2(X), d5(X), d1(X).
d0(X) :- d9(X), d3(X), not d8(X).
d7(X) :- d9(X), not d3(X), d2(X).
d2(X) :- d6(X), not d1(X), d4(X).
d4(X) :- d0(X), d6(X), not d3(X).
d3(X) :- d9(X), not d7(X), d4(X).
d1(X) :- d3(X), not d5(X), d6(X).
d6(X) :- not d8(X), d7(X), d4(X).
d2(X) :- not d8(X), d3(X), d7(X).
d4(X) :- d9(X), not d7(X), d8(X).