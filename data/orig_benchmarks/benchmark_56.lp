d1(o0).
d3(o6).
d3(o3).
d1(o8).
d0(o9).
d4(o9).

d2(X) :- d3(X), not d1(X).
d2(X) :- not d0(X), d4(X).
d2(X) :- d0(X), not d1(X).
d5(X) :- not d3(X), d4(X).
d4(X) :- d0(X), not d3(X).
d1(X) :- not d2(X), d4(X).
d1(X) :- d0(X), not d5(X).
d1(X) :- d0(X), not d3(X).
d4(X) :- d2(X), not d0(X).
d2(X) :- d0(X), not d1(X).
d5(X) :- not d1(X), d4(X).