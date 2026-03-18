d9(o5).
d1(o5).
d2(o0).
d8(o0).
d4(o2).
d1(o3).

d9(X) :- not d2(X), d1(X).
d5(X) :- d0(X), not d7(X).
d3(X) :- d1(X), not d0(X).
d7(X) :- d6(X), not d1(X).
d6(X) :- d9(X), not d2(X).
d0(X) :- not d9(X), d4(X).
d7(X) :- not d2(X), d4(X).
d7(X) :- d5(X), not d9(X).
d0(X) :- d3(X), not d1(X).
d7(X) :- not d3(X), d4(X).
d7(X) :- not d2(X), d0(X).