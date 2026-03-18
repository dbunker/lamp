d9(o4).
d1(o4).
d5(o5).
d5(o0).
d3(o3).
d3(o5).
d6(o5).
d1(o3).
d6(o2).

d3(X) :- d2(X), not d1(X), not d4(X).
d2(X) :- d9(X), not d8(X), not d0(X).
d9(X) :- not d3(X), not d5(X), d4(X).
d0(X) :- not d2(X), not d5(X), d8(X).
d4(X) :- d2(X), not d7(X), not d0(X).
d7(X) :- not d2(X), d0(X), not d9(X).
d7(X) :- not d2(X), not d8(X), d4(X).
d8(X) :- not d5(X), d7(X), not d4(X).
d9(X) :- not d3(X), not d6(X), d8(X).
d0(X) :- not d6(X), not d5(X), d1(X).
d1(X) :- not d3(X), not d6(X), d8(X).