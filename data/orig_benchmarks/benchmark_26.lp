d8(o2).
d5(o5).
d0(o0).
d4(o0).
d5(o4).
d2(o4).
d9(o2).
d5(o2).
d7(o5).

d9(X) :- d2(X), not d4(X).
d3(X) :- not d8(X), d4(X).
d6(X) :- d7(X), not d1(X).
d8(X) :- d3(X), not d4(X).
d2(X) :- d3(X), not d4(X).
d9(X) :- not d3(X), d7(X).
d1(X) :- not d3(X), d6(X).
d1(X) :- d7(X), not d4(X).
d0(X) :- d2(X), not d8(X).
d1(X) :- not d2(X), d8(X).
d3(X) :- d7(X), not d4(X).