d6(o2).
d5(o4).
d0(o4).
d4(o9).
d7(o2).
d4(o4).
d1(o6).
d1(o7).
d7(o5).

d8(X) :- d5(X), not d0(X), not d9(X).
d8(X) :- d2(X), not d4(X), not d9(X).
d9(X) :- not d8(X), d6(X), not d5(X).
d4(X) :- not d3(X), d5(X), not d0(X).
d1(X) :- not d3(X), not d6(X), d4(X).
d4(X) :- not d3(X), d2(X), not d5(X).
d2(X) :- not d8(X), d3(X), not d9(X).
d3(X) :- not d1(X), d8(X), not d4(X).
d0(X) :- d6(X), not d1(X), not d7(X).
d0(X) :- d9(X), not d7(X), not d4(X).
d8(X) :- d6(X), not d5(X), not d0(X).