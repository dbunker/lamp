d5(o7).
d6(o7).
d2(o9).
d9(o2).
d7(o7).
d1(o7).

d8(X) :- d0(X), not d3(X).
d0(X) :- not d2(X), d5(X).
d1(X) :- d2(X), not d8(X).
d5(X) :- not d3(X), d7(X).
d4(X) :- not d8(X), d3(X).
d3(X) :- d2(X), not d0(X).
d9(X) :- not d3(X), d2(X).
d0(X) :- d6(X), not d1(X).
d7(X) :- d2(X), not d9(X).
d9(X) :- d0(X), not d5(X).
d8(X) :- not d2(X), d3(X).