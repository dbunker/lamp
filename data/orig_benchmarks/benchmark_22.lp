d1(o5).
d2(o0).
d5(o4).
d0(o4).
d8(o5).
d4(o2).
d6(o5).
d0(o3).
d7(o5).

d2(X) :- d3(X), d5(X), not d1(X).
d0(X) :- d9(X), d3(X), not d7(X).
d9(X) :- not d8(X), d3(X), d5(X).
d3(X) :- d9(X), not d1(X), d8(X).
d7(X) :- not d3(X), d6(X), d4(X).
d4(X) :- d9(X), not d6(X), d5(X).
d8(X) :- not d2(X), d0(X), d5(X).
d7(X) :- d0(X), d8(X), not d9(X).
d2(X) :- d0(X), not d5(X), d4(X).
d2(X) :- not d3(X), d6(X), d1(X).
d5(X) :- d2(X), d7(X), not d1(X).