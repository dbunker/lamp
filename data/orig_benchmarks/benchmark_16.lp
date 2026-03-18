d4(o5).
d3(o6).
d4(o4).
d6(o3).
d4(o8).
d0(o2).

d4(X) :- not d7(X), d5(X).
d5(X) :- d9(X), not d4(X).
d8(X) :- d6(X), not d4(X).
d5(X) :- not d2(X), d0(X).
d9(X) :- d3(X), not d4(X).
d2(X) :- not d5(X), d4(X).
d8(X) :- d0(X), not d4(X).
d1(X) :- d7(X), not d0(X).
d7(X) :- not d8(X), d3(X).
d4(X) :- d7(X), not d0(X).
d4(X) :- d5(X), not d9(X).