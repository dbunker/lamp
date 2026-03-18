d9(o0).
d0(o5).
d5(o1).
d0(o1).
d3(o4).
d7(o3).

d6(X) :- not d3(X), d5(X), not d1(X).
d9(X) :- not d3(X), not d0(X), d4(X).
d2(X) :- not d8(X), d6(X), not d9(X).
d7(X) :- not d2(X), d0(X), not d6(X).
d8(X) :- d7(X), not d4(X), not d0(X).
d2(X) :- not d3(X), d6(X), not d0(X).
d1(X) :- not d6(X), not d7(X), d8(X).
d0(X) :- not d5(X), not d7(X), d4(X).
d2(X) :- not d9(X), not d5(X), d4(X).
d6(X) :- not d3(X), not d8(X), d4(X).
d4(X) :- not d8(X), d6(X), not d0(X).